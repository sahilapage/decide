"""
Hand-written "foreign" test set: tickets, products and help articles unlike the
training templates, with edits the training data never uses. For evaluation
only (never trained on). Labels follow rules visible in the training data:

  approve  amount <= half the plan's cap and no red flag -> True (88% of such
           training rows are approved); amount >= 1.5x cap or a red flag -> False
  grounded supported only if every claim in the draft is in the article
  route    the queue a support lead would pick (no label noise)

  python tests/make_foreign_set.py      # writes tests/foreign_set.jsonl
  python test.py tests/foreign_set.jsonl
"""

import json
import os

# (subject, body, route, tier, amount, approve, tag)
TICKETS = [
    ("Double charged for annual renewal", "Hey team, my card got hit twice for the yearly renewal on Oct 2. Can one of those be reversed?", "billing", "pro", 49.0, True, "plain"),
    ("Invoice shows wrong VAT", "Our invoice INV-88213 lists 20% VAT but we are VAT-registered in Ireland and should be reverse-charged. Please reissue it.", "billing", "enterprise", 120.0, True, "plain"),
    ("refund pls", "upgraded to pro by mistake yesterday, didnt mean to. can i get the money back", "billing", "pro", 29.0, True, "typos"),
    ("Charged after cancelling", "I cancelled on the 3rd (confirmation CX-4410) yet I was billed again on the 10th. I'd like that refunded.", "billing", "free", 9.99, True, "plain"),
    ("Promo code not applied", "I used code AUTUMN20 at checkout but the full price was taken. Could you refund the difference?", "billing", "free", 3.0, True, "plain"),
    ("Unauthorised charge", "There's a charge on my statement I don't recognise. If this isn't sorted today I'm calling my bank to claw it back.", "billing", "pro", 40.0, False, "redflag-paraphrase"),
    ("Price went up", "My monthly price jumped from $12 to $15 with no email. I know it's against the rules but could you do me a favour just this once and keep the old rate?", "billing", "free", 3.0, False, "redflag-paraphrase"),
    ("Seats + small bug", "Main thing: I was charged for 6 seats but we only have 4 users. Separately, less urgent: the export button is greyed out sometimes.", "billing", "enterprise", 150.0, True, "multi-issue"),
    ("App crashes on launch", "Since the 4.2 update the Android app closes immediately when I open it. Reinstalled twice, same thing. Pixel 7, Android 14.", "technical", "pro", 300.0, False, "plain"),
    ("API returns 429", "Our integration keeps getting HTTP 429 Too Many Requests even though we're well under 100 calls a minute. Started this morning.", "technical", "enterprise", 1200.0, False, "plain"),
    ("SSO login loop", "When our staff sign in with Okta SSO they get bounced back to the login page in a loop. Cleared cookies, no luck.", "technical", "enterprise", 90.0, True, "plain"),
    ("CSV export broken", "Exports to CSV come out with every column shifted by one. This is breaking our weekly report 😩", "technical", "free", 60.0, False, "emoji"),
    ("webhook not firing", "the order.created webhook stopped hitting our endpoint around 2am UTC. our server logs show nothing incoming.", "technical", "pro", 20.0, True, "plain"),
    ("Dark mode text unreadable", "In dark mode the table headers are black on dark grey, basically invisible. Chrome 129 on Windows 11.", "technical", "free", 5.0, True, "plain"),
    ("NOTHING WORKS", "THE DASHBOARD HAS BEEN BLANK FOR TWO DAYS. WE PAY FOR THIS. FIX IT.", "technical", "enterprise", 2000.0, False, "caps"),
    ("Re: Re: sync issue", "Hi again,\n\n> On Monday, support wrote: Could you try logging out and back in?\n\nI did that, the calendar sync still stops after a few minutes. Any other ideas?\n\nThanks, Priya", "technical", "pro", 25.0, True, "thread"),
    ("Can't reset password", "The password reset email never arrives. Checked spam. My username is jdoe_77.", "account", "free", 4.0, True, "plain"),
    ("Change account owner", "Our admin left the company. How do we transfer ownership of the workspace to me?", "account", "enterprise", 700.0, False, "plain"),
    ("Lost 2FA phone", "I lost my phone and can't get the 2FA codes any more. How can I get back into my account?", "account", "pro", 10.0, True, "plain"),
    ("Delete my account", "Please delete my account and all associated data under GDPR.", "account", "free", 45.0, False, "plain"),
    ("Update login email", "I need to change the login email from my old work address to my personal one.", "account", "pro", 35.0, True, "plain"),
    ("Two accounts", "I made this account only last week and accidentally created a second one with another email. Can they be merged? Also refund the duplicate please.", "account", "free", 8.0, False, "redflag-paraphrase"),
    ("Account suspended?", "I got a message saying my account is suspended for suspicious activity. I haven't done anything unusual.", "account", "enterprise", 100.0, True, "plain"),
    ("acount hacked??", "someone chnaged my pasword and email, i cant log in anymore, plz help asap", "account", "pro", 250.0, False, "typos"),
    ("Parcel stuck in customs", "My order 55120 has been sitting in customs in Toronto for 9 days. Is there anything you can do?", "shipping", "pro", 18.0, True, "plain"),
    ("Wrong address on order", "I just noticed the shipping address on order 77341 is my old flat. Can it still be changed?", "shipping", "free", 12.0, True, "plain"),
    ("Box arrived damaged", "The box was crushed and the lamp inside is cracked. Photos attached. Order 30982.", "shipping", "enterprise", 140.0, True, "plain"),
    ("Missing item", "Ordered 3 mugs, only 2 in the package. Packing slip says 3.", "shipping", "free", 6.0, True, "plain"),
    ("Marked delivered, not received", "Tracking says delivered yesterday at 4pm but there is nothing at my door or with the neighbours.", "shipping", "pro", 54.0, True, "plain"),
    ("Express was slow", "I paid for express (2-day) and it took 6 days. Can I get the express fee back?", "shipping", "pro", 15.0, True, "plain"),
    ("Replacement broken too", "This is not the first time I'm writing about this: the replacement headphones also arrived with a broken hinge.", "shipping", "enterprise", 80.0, False, "redflag-literal"),
    ("PO box in Alaska", "Do you deliver to PO boxes in Alaska? I need it there by next month.", "shipping", "free", 70.0, False, "plain"),
    ("Feature request: Gantt view", "Would love a Gantt chart view for projects. Any plans for that?", "general", "pro", 400.0, False, "plain"),
    ("iPad support", "Just wondering if your app supports iPad with split screen.", "general", "free", 2.0, True, "plain"),
    ("Partnership inquiry", "I run a small design agency and would like to resell your product. Who should I talk to?", "general", "enterprise", 1000.0, False, "plain"),
    ("Thanks!", "Just wanted to say the new update is great, search is so much faster. Keep it up!", "general", "pro", 5.0, True, "plain"),
    ("Accessibility", "Does the web app work with screen readers like NVDA?", "general", "enterprise", 60.0, True, "plain"),
    ("Data residency", "Where are your servers located? We need to know whether our data stays in the EU.", "general", "free", 100.0, False, "plain"),
    ("Holiday hours", "Is support open over the Christmas holidays?", "general", "pro", 500.0, False, "plain"),
    ("Pregunta sobre integración", "Hola, does your product integrate with Shopify? Gracias!", "general", "free", 10.0, True, "mixed-language"),
]

LONG_KB = ("Our status page lists every incident within 15 minutes of detection. Planned maintenance is announced 72 hours ahead. "
           "Maintenance windows run on Sundays between 1am and 3am UTC. During maintenance the API may return HTTP 503. "
           "Enterprise customers receive incident reports within 5 business days. Service credits are issued automatically "
           "when monthly uptime falls below 99.9%. Credits are applied to the next invoice.")

# (kb_article, draft_reply, grounded, tag)
PAIRS = [
    ("Annual plans can be refunded in full within 30 days of purchase. After 30 days, refunds are prorated for unused months.",
     "You're within the first 30 days, so the annual plan qualifies for a full refund.", True, "paraphrase"),
    ("Annual plans can be refunded in full within 30 days of purchase. After 30 days, refunds are prorated for unused months.",
     "Annual plans can be fully refunded within 60 days of purchase.", False, "number"),
    ("Invoices can be reissued with corrected tax details from Settings > Billing. Reissued invoices keep the original number with an -R suffix.",
     "You can reissue the invoice with the right tax details under Settings > Billing; it keeps the same number with -R at the end.", True, "paraphrase"),
    ("Invoices can be reissued with corrected tax details from Settings > Billing. Reissued invoices keep the original number with an -R suffix.",
     "You can reissue it with corrected tax details from the Invoices tab in the mobile app.", False, "place"),
    ("Support is available Monday to Friday, 9am to 6pm CET, except public holidays.",
     "Our team is around Monday to Friday from 9am to 6pm CET, apart from public holidays.", True, "paraphrase"),
    ("Support is available Monday to Friday, 9am to 6pm CET, except public holidays.",
     "Our team is available Monday to Saturday, 9am to 6pm CET.", False, "day"),
    ("Workspace ownership can be transferred by the current owner or, if the owner has left, by any admin after identity verification.",
     "Since your owner has left, an admin can transfer ownership once we verify their identity.", True, "paraphrase"),
    ("Workspace ownership can be transferred by the current owner or, if the owner has left, by any admin after identity verification.",
     "Since your owner has left, any member of the workspace can transfer ownership once we verify their identity.", False, "role"),
    ("Accounts requested for deletion are permanently erased within 30 days, including backups.",
     "Once you request deletion, your account and its backups are erased for good within 30 days.", True, "paraphrase"),
    ("Accounts requested for deletion are permanently erased within 30 days, including backups.",
     "Once you request deletion, your account and its backups are erased within 7 days.", False, "number"),
    ("Express orders ship the same day if placed before 2pm local time and arrive in 2 business days.",
     "Express orders placed before 2pm go out the same day and arrive in 2 business days.", True, "paraphrase"),
    ("Express orders ship the same day if placed before 2pm local time and arrive in 2 business days.",
     "Express orders placed before 5pm go out the same day and arrive in 2 business days.", False, "time"),
    ("Damaged items can be returned for a full refund if reported within 14 days of delivery with photos.",
     "Since you reported the damage within 14 days and sent photos, you're eligible for a full refund.", True, "paraphrase"),
    ("Damaged items can be returned for a full refund if reported within 14 days of delivery with photos.",
     "You can return any item for a full refund within 14 days of delivery.", False, "condition-dropped"),
    ("The API allows 100 requests per minute per key. Requests over the limit receive HTTP 429 and can be retried after 60 seconds.",
     "Each key can make 100 requests a minute; above that you get a 429 and can retry after 60 seconds.", True, "paraphrase"),
    ("The API allows 100 requests per minute per key. Requests over the limit receive HTTP 429 and can be retried after 60 seconds.",
     "Each key can make 500 requests a minute; above that you get a 429 and can retry after 60 seconds.", False, "number"),
    ("Two-factor authentication can be reset by contacting support from the account's registered email address. Resets take up to 48 hours.",
     "Email us from your registered address and we'll reset your 2FA within two days.", True, "cross-unit"),
    ("Two-factor authentication can be reset by contacting support from the account's registered email address. Resets take up to 48 hours.",
     "Call our phone line and we'll reset your 2FA within 48 hours.", False, "channel"),
    ("Gift cards never expire and can be used on any product except subscriptions.",
     "Gift cards don't expire, and you can use them on anything other than subscriptions.", True, "paraphrase"),
    ("Gift cards never expire and can be used on any product except subscriptions.",
     "Gift cards don't expire, and you can use them on anything, including subscriptions.", False, "scope"),
    ("Orders shipped to Canada may be held by customs for up to 10 business days. Duties are paid by the recipient.",
     "Customs in Canada can hold parcels for up to 10 business days, and the recipient pays any duties.", True, "paraphrase"),
    ("Orders shipped to Canada may be held by customs for up to 10 business days. Duties are paid by the recipient.",
     "Customs in Canada can hold parcels for up to 10 business days, and we cover any duties.", False, "who-pays"),
    ("Data for EU customers is stored in Frankfurt, Germany, and is never transferred outside the EU.",
     "As an EU customer, your data stays in our Frankfurt data centre and never leaves the EU.", True, "paraphrase"),
    ("Data for EU customers is stored in Frankfurt, Germany, and is never transferred outside the EU.",
     "As an EU customer, your data stays in our Dublin data centre and never leaves the EU.", False, "place"),
    ("Nonprofits get 30% off any annual plan after verification.",
     "Once you're verified as a nonprofit, you'll get 30% off an annual plan.", True, "paraphrase"),
    ("Nonprofits get 30% off any annual plan after verification.",
     "Once you're verified as a nonprofit, you'll get 50% off an annual plan.", False, "number"),
    ("Seats are billed monthly based on the number of active users on the 1st of each month.",
     "We bill seats each month based on how many users are active on the 1st.", True, "paraphrase"),
    ("Seats are billed monthly based on the number of active users on the 1st of each month.",
     "We bill seats each month based on how many users are active on the 15th.", False, "date"),
    ("Promo codes must be entered before payment; they cannot be applied to completed orders.",
     "Promo codes need to be entered before you pay, so they can't be added to an order that's already complete.", True, "paraphrase"),
    ("Promo codes must be entered before payment; they cannot be applied to completed orders.",
     "We can apply the promo code to your completed order retroactively.", False, "contradiction"),
    (LONG_KB, "Service credits are issued automatically when monthly uptime falls below 99.9%, and they're applied to your next invoice.", True, "long-kb"),
    (LONG_KB, "Planned maintenance is announced 24 hours ahead and runs on Sundays between 1am and 3am UTC.", False, "long-kb-number"),
    ("Replacements are shipped within 3 business days.", "We'll ship your replacement within 3 business days.", True, "one-sentence-kb"),
    ("Replacements are shipped within 3 business days.", "We'll ship your replacement within 3 business days and add a free gift card.", False, "extra-promise"),
    ("Price changes are announced by email at least 30 days before they take effect.",
     "You'll always hear from us by email a month or more before a new price applies.", True, "paraphrase-unit"),
    ("Price changes are announced by email at least 30 days before they take effect.",
     "Price changes take effect immediately and are announced in the app.", False, "contradiction"),
    ("Screen readers are supported: NVDA and JAWS on Windows, and VoiceOver on macOS.",
     "Yes, it works with NVDA and JAWS on Windows, and with VoiceOver on macOS.", True, "paraphrase"),
    ("Screen readers are supported: NVDA and JAWS on Windows, and VoiceOver on macOS.",
     "Yes, it works with NVDA and JAWS on Windows and on Linux.", False, "platform"),
    ("PO boxes are supported for standard shipping within the US, including Alaska and Hawaii. Express shipping requires a street address.",
     "We can ship to a PO box in Alaska with standard shipping; express needs a street address.", True, "paraphrase"),
    ("PO boxes are supported for standard shipping within the US, including Alaska and Hawaii. Express shipping requires a street address.",
     "We can ship to a PO box in Alaska with express shipping.", False, "condition-swapped"),
]


def main():
    assert len(TICKETS) == len(PAIRS) == 40
    here = os.path.dirname(os.path.abspath(__file__))
    tags = {}
    # exactly the dev.jsonl format: request_id, state (6 fields, same order), labels (3 fields)
    with open(os.path.join(here, "foreign_set.jsonl"), "w") as fh:
        for i, ((subj, body, route, tier, amt, appr, ttag), (kb, draft, gr, gtag)) in enumerate(zip(TICKETS, PAIRS)):
            rid = "req-%06d" % (900001 + i)
            tags[rid] = {"ticket": ttag, "grounded": gtag}
            fh.write(json.dumps({
                "request_id": rid,
                "state": {"subject": subj, "body": body, "customer_tier": tier, "amount": float(amt),
                          "draft_reply": draft, "kb_article": kb},
                "labels": {"route": route, "approve": appr, "grounded": gr}}) + "\n")
    json.dump(tags, open(os.path.join(here, "foreign_set_tags.json"), "w"), indent=1)
    print("wrote tests/foreign_set.jsonl and tests/foreign_set_tags.json")


if __name__ == "__main__":
    main()
