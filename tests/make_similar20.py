"""
20 new hand-written tickets in the same style and difficulty as dev.jsonl:
same products (Corvus, Quill), same kinds of help articles (cancellations,
exports, password resets, delivery, warranty, price notices) and the same
kinds of draft edits (audience, number, monthly/annual, channel, format).
None is a copy of a dev or train row. Same format as dev.jsonl.

Labels: approve True when the amount is at most half the plan's cap with no
red flag, False at 1.5x the cap or with a red flag (the rule visible in the
training data); grounded True only if every claim in the draft is in the
article; route is the ticket's main issue.

  python tests/make_similar20.py          # writes tests/similar20.jsonl
  python test.py tests/similar20.jsonl
"""

import json
import os

KB_CANCEL_ANNUAL = ("Cancellations can be made up to fourteen days before renewal by annual subscribers. Access continues "
                    "until the end of the current billing period, and there is no cancellation fee.")
KB_EXPORT_ADMIN = ("Exports are available to administrators. Links remain valid for seven days and arrive as CSV only. "
                   "Large exports take up to 24 hours to prepare.")
KB_EXPORT_ALL = ("Exports are available to all users. Links remain valid for 14 days and arrive as CSV and JSON. Large "
                 "exports take up to 24 hours to prepare.")
KB_RESET_EMAIL = ("Password resets are available to all users via an email link. Links expire after 60 minutes. Accounts "
                  "lock automatically after five failed attempts.")
KB_RESET_ADMIN = "Password resets are available to administrators via recovery codes. A reset link is valid for thirty minutes."
KB_RESET_SMS = "Password resets are available to all users via an SMS code. Links expire after 60 minutes."
KB_WARRANTY = ("We warrant every delivery for 365 days after delivery. Accepted claims receive a full refund, and we "
               "require photographs of the fault to open one.")
KB_DELIVERY = "Standard delivery takes 7 days for domestic orders. Remote addresses may add 3 days."

# (subject, body, tier, amount, route, approve, draft, kb, grounded)
ROWS = [
    ("charged twice this month",
     "Hello, I was billed twice for the same subscription period on my card. Could you refund the duplicate? Thanks. I'm on the Pro plan. The amount in question is $18.40.",
     "pro", 18.40, "billing", True,
     "I can see the duplicate - refunds go back to the original payment method within five business days.",
     "Duplicate charges are refunded to the original payment method within 5 business days.", True),
    ("refund for annual plan",
     "Hi, I'd like a refund for the annual plan I bought last week, it doesn't suit our team. Appreciate the help. I'm on the Free plan.",
     "free", 64.10, "billing", False,
     "Cancelling is available to all subscribers with 14 days notice, and access continues until the end of the billing period with no cancellation fee.",
     KB_CANCEL_ANNUAL, False),
    ("invoice question",
     "Our latest invoice lists 12 seats but we only have 10 active users. Could you correct it and refund the difference? Regards.",
     "enterprise", 95.30, "billing", True,
     "monthly subscribers hear about price changes thirty days ahead by email, and the new price starts from the next billing cycle.",
     "We notify monthly subscribers of pricing changes 30 days in advance by email, effective from the following billing cycle.", True),
    ("charged after cancelling",
     "I cancelled last month but was charged again anyway. I raised this before and nothing happened. The amount in question is $42.15.",
     "pro", 42.15, "billing", False,
     "Queued. all users get exports in CSV only; the link is valid 7 days and it takes up to 24 hours to build.",
     KB_EXPORT_ADMIN, False),
    ("export stuck",
     "The export has been preparing for two days now and never finishes. I have already tried logging out and back in. Thanks in advance.",
     "free", 6.20, "technical", True,
     "Your export is queued - all users can run these, files come as CSV and JSON, the link lasts two weeks and it can take up to 24 hours.",
     KB_EXPORT_ALL, True),
    ("calendar sync not working",
     "Calendar sync stopped working for our whole team after yesterday's update. We've tried reconnecting the integration twice. The amount in question is $820.50.",
     "enterprise", 820.50, "technical", False,
     "I've sent a reset link by email - it expires in an hour, and the account locks after ten failed attempts.",
     KB_RESET_EMAIL, False),
    ("save button does nothing",
     "Clicking save in the report builder does nothing, no error message at all. Tried Firefox and Chrome. I'm on the Pro plan.",
     "pro", 25.60, "technical", True,
     "Items are covered for a year from delivery, so with photos of the fault I can open a claim for a full refund.",
     KB_WARRANTY, True),
    ("Two-factor codes not arriving",
     "The SMS codes for two-factor never arrive on my phone. I've waited ten minutes each time. Regards.",
     "free", 3.80, "technical", True,
     "I've triggered a reset with your recovery codes - that's available to administrators, and the link is valid for 30 minutes.",
     KB_RESET_ADMIN, True),
    ("change account email",
     "Please change the login email on my account, I no longer have access to the old address. Thanks.",
     "pro", 12.40, "account", True,
     "Reset sent via an SMS code. That route is for administrators only; the link lasts 60 minutes.",
     KB_RESET_SMS, False),
    ("add a new admin",
     "We need to give admin rights to a new team member, how do we do that? Our previous admin has left the company.",
     "enterprise", 150.25, "account", True,
     "Monthly subscribers can cancel up to thirty days before renewal, and there's no fee for cancelling.",
     "Cancellations can be made up to 30 days before renewal by monthly subscribers. There is no cancellation fee.", True),
    ("can't log in",
     "My password isn't accepted even though I'm sure it's right. I only signed up recently. Please advise.",
     "free", 9.10, "account", False,
     "Domestic orders usually arrive in 5 days and are dispatched within 2 business days.",
     "Standard delivery takes 5 days for domestic orders. Orders are dispatched within 2 business days.", True),
    ("delete my workspace",
     "Please close our workspace and delete all data, we've moved to another tool. Thanks.",
     "pro", 260.45, "account", False,
     "Before we close it, you can export everything - the link stays live for a month and files come as CSV and JSON.",
     "Exports are available to all users. Links remain valid for 30 days and arrive as JSON only.", False),
    ("Package hasn't arrived",
     "Hi, Order #20417 was due last week and tracking hasn't moved for six days. Can you check? Thanks.",
     "free", 11.75, "shipping", True,
     "Domestic orders normally take 7 days, and remote addresses can add 3 days.",
     KB_DELIVERY, True),
    ("Wrong colour sent",
     "Order #20533 arrived in the wrong colour. I need the blue one I actually ordered. Thanks in advance. I'm on the Pro plan.",
     "pro", 33.20, "shipping", True,
     "Reports inside 14 days are covered for domestic orders only, so I can arrange a replacement once you confirm.",
     "all orders are covered if damage is reported inside fourteen days. We will organise a replacement or refund, your choice, with replacements dispatched within 3 days.", False),
    ("Damaged on arrival",
     "Order #20611 arrived with the box crushed and the screen cracked. Photos attached.",
     "enterprise", 640.90, "shipping", False,
     "We warrant deliveries for six months, so you qualify for a full refund with photographs of the fault.",
     KB_WARRANTY, False),
    ("item missing from order",
     "Order #20702 is missing one of the two items. If this isn't fixed I will file a chargeback with my bank.",
     "free", 4.95, "shipping", False,
     "Your parcel goes out with an express partner within 1 business day.",
     "Orders are dispatched via an express partner within 1 business day. Standard delivery takes 7 days.", True),
    ("question about Quill",
     "Does Quill support scheduled runs for reports? Just checking before we upgrade.",
     "pro", 7.30, "general", True,
     "annual subscribers get two weeks notice by email before a price change.",
     "We notify annual subscribers of pricing changes fourteen days in advance via an in-app notice.", False),
    ("feature idea",
     "It would be great if Corvus could export dashboards to PDF. Any plans for that? Regards.",
     "free", 90.60, "general", False,
     "Exports are open to all users, arrive as CSV and JSON, and the link stays valid for 14 days.",
     KB_EXPORT_ALL, True),
    ("API rate limits",
     "What are the API rate limits on the Enterprise plan? We're planning a bulk integration next quarter.",
     "enterprise", 120.35, "general", True,
     "annual subscribers can cancel up to fourteen days before renewal, and access runs until the end of the billing period.",
     KB_CANCEL_ANNUAL, True),
    ("multiple questions",
     "Some context first: we're a team of eight. Most importantly: does Corvus have a mobile app? Thanks.",
     "pro", 15.80, "general", True,
     "Reset sent via an SMS code - all users can use it, and it expires after ten minutes.",
     KB_RESET_SMS, False),
]


def main():
    assert len(ROWS) == 20
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "similar20.jsonl")
    with open(path, "w") as fh:
        for i, (subj, body, tier, amt, route, appr, draft, kb, gr) in enumerate(ROWS):
            fh.write(json.dumps({
                "request_id": "req-%06d" % (950001 + i),
                "state": {"subject": subj, "body": body, "customer_tier": tier, "amount": float(amt),
                          "draft_reply": draft, "kb_article": kb},
                "labels": {"route": route, "approve": appr, "grounded": gr}}) + "\n")
    print("wrote tests/similar20.jsonl")


if __name__ == "__main__":
    main()
