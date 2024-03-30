from twilio.rest import Client

account_sid = 'AC735fdcab4a0f13d673905ddd3ba73018'
auth_token = '873d5a093aae2c37a8d7488785b270f1'
client = Client(account_sid, auth_token)

call = client.calls.create(
    twiml='<Response><Say>Emergency Fall Detected please respond Thank you</Say></Response>',
    to='+919860215374',
    from_='+15627845845'
)

print(call.sid)