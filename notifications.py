from twilio.rest import Client
account_sid = 'ACcc51369f0edfe1dee4b2e5c5e37d95d9'
auth_token = '51a9fb632cd1be08088f1c47c42c6dd3'
def make_phone_call():
    
    client = Client(account_sid, auth_token)

    call = client.calls.create(
        twiml='<Response><Say>Emergency Fall Detected please respond Thank you</Say></Response>',
        to='+919370417601',
        from_='+19285506087'
    )

    print(call.sid)


def whatsapp_message():
 
    # account_sid = 'ACf535dc7cbe7f698fb66f33e7f962b899' 
    # auth_token = '45cff229242d4fe9bd704739e4d8b3ec' 
    client = Client(account_sid, auth_token) 
    message = client.messages.create( 
                                body='Alert !! Fall detected !!!  ',
                                from_='whatsapp:+14155238886',   
                                media_url='https://013f-2401-4900-502e-f6c9-512-9b0f-39ae-f9e5.ngrok-free.app/tmp/frame0.jpg',     
                                to='whatsapp:+919370417601' 
                            ) 
    print(message)
    print(message.sid)