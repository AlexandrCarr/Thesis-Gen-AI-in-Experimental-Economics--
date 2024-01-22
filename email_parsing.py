from __future__ import print_function

import re

from response_generation import respond

import os.path
import base64
import time
import email
import random

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']  # .readonly would forbid editing


def authenticate():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return creds


def extract_html_payload(mime_msg):
    html = ""
    message_main_type = mime_msg.get_content_maintype()
    if message_main_type == 'multipart':
        for part in mime_msg.get_payload():
            if part.get_content_maintype() == 'text':
                html += part.get_payload()
    elif message_main_type == 'text':
        html += mime_msg.get_payload()

    return html


def get_job_category(html):
    if "Od 0 K=C4=8D, Zkr" in html:
        return "part_time"
    elif "Od 50 000 K=C4=8D, Pln=C3=BD =C3=BAvazek" in html:
        return "high_income"
    elif "Od 0 K=C4=8D, Pln=C3=BD =C3=BAvazek" in html:
        return "low_income"
    else:
        return "not_offer_email"


def respond_unread_emails():
    try:
        # Call the Gmail API
        creds = authenticate()
        service = build('gmail', 'v1', credentials=creds)

        # Reads
        results = service.users().messages().list(userId='me',
                                                  labelIds=['INBOX'],
                                                  q="is:unread").execute()
        messages = results.get("messages", [])

        if messages:  # if some messages were found
            for i, message in enumerate(messages):  # Goes over all unread messages

                # Read the message body
                msg = service.users().messages().get(userId='me',
                                                            id=message['id'],
                                                            format="raw").execute()
                # Mark the email as read to not go over it twice
                service.users().messages().modify(userId='me',
                                                 id=message['id'],
                                                 body={'removeLabelIds': ['UNREAD']}).execute()

                # Decode the message
                mime_msg = email.message_from_bytes(base64.urlsafe_b64decode(msg['raw']))
                html = extract_html_payload(mime_msg)

                # Extract relevant urls with job offers
                pattern = r'"(http://tracking\.jobs\.cz[^"]*)"'
                job_urls = re.findall(pattern, html)
                job_urls = [url.strip().replace(" ", "")
                            .replace("=", "")
                            .replace("\r\n", "") for url in job_urls]

                category = get_job_category(html)  # Classify the offer into 1 of 5 types

                if category == "not_offer_email":  # Continue if it's not
                    print("not an email with offer")
                    continue

                # Respond to each job offering
                for url in job_urls:
                    respond(url, category)

                    delay = random.uniform(2.0, 10.0)
                    print("Sleep " + str(delay) + "s")
                    time.sleep(delay)

                    print("url resolved")

                print("ALL EMAILS RESOLVED, NO MORE PENDING OFFERS AT THIS MOMENT")
                time.sleep(1)
        else:
            print("no new messages")
            pass

    except HttpError as error:
        # TODO(developer) - Handle errors from gmail API.
        print(f'An error occurred: {error}')


if __name__ == '__main__':
    respond_unread_emails()