import os
import pickle
import logging
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from datetime import datetime, timedelta

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

class CalendarModule:
    def __init__(self, credentials_path='credentials.json', token_path='token.pickle'):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None
        self.logger = logging.getLogger("CalendarModule")

    def authenticate(self):
        """Authenticates with Google Calendar API."""
        creds = None
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    self.logger.error(f"Error refreshing credentials: {e}")
                    creds = None
            
            if not creds:
                if not os.path.exists(self.credentials_path):
                    return False, f"Credentials file missing: {self.credentials_path}"
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open(self.token_path, 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('calendar', 'v3', credentials=creds)
        return True, "Authenticated successfully."

    def get_upcoming_meets(self, hours=24):
        """Fetches upcoming Google Meet events within the next specified hours."""
        if not self.service:
            success, msg = self.authenticate()
            if not success:
                return []

        now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
        then = (datetime.utcnow() + timedelta(hours=hours)).isoformat() + 'Z'
        
        try:
            events_result = self.service.events().list(
                calendarId='primary', timeMin=now, timeMax=then,
                singleEvents=True, orderBy='startTime').execute()
            events = events_result.get('items', [])
            
            meet_events = []
            for event in events:
                if 'conferenceData' in event or 'hangoutLink' in event:
                    meet_events.append({
                        'summary': event.get('summary', 'No Title'),
                        'start': event['start'].get('dateTime', event['start'].get('date')),
                        'link': event.get('hangoutLink', '')
                    })
            return meet_events
        except Exception as e:
            self.logger.error(f"Error fetching calendar events: {e}")
            return []

    def check_for_reminders(self):
        """Checks for upcoming Meets and returns alerts if they are starting soon (e.g., in 5 mins)."""
        upcoming = self.get_upcoming_meets(hours=1)
        alerts = []
        now = datetime.utcnow()
        for meet in upcoming:
            start_time = datetime.fromisoformat(meet['start'].replace('Z', ''))
            time_diff = start_time - now
            if timedelta(minutes=0) <= time_diff <= timedelta(minutes=5):
                alerts.append(f"Meeting '{meet['summary']}' starting in {int(time_diff.total_seconds() / 60)} minutes. Link: {meet['link']}")
        return alerts
