from smtplib import SMTP

#Use Google SMTP server
s = smtplib.SMTP('smtp.gmail.com', 587)
#s.ehlo()  
# start TLS for security 
s.starttls() 

# Authentication 
s.login("Email_Address@gmail.com", "Email_Password") 
  
# message to be sent 
message = "Message_you_need_to_send"
  
# sending the mail 
s.sendmail("From_Address@gmail.com", "To_Address@gmail.com", message) 
  
# terminating the session 
s.quit() 
  
#Resources: 
#  (1) How to send mail using Python:https://www.geeksforgeeks.org/send-mail-gmail-account-using-python/
#  (2) Error Messages: https://stackoverflow.com/questions/26852128/smtpauthenticationerror-when-sending-mail-using-gmail-and-python 
#  (3) https://docs.python.org/3/library/smtplib.html
