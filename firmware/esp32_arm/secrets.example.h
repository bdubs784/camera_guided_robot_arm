// Copy this file to secrets.h (same folder) and fill in your own values.
// secrets.h is listed in .gitignore, so it never gets committed.
#pragma once
#define SECRET_HOME_SSID     "your-wifi-name"
#define SECRET_HOME_PASSWORD "your-wifi-password"
#define SECRET_AP_SSID       "ArmControl"        // the arm's own network if home WiFi is not found
#define SECRET_AP_PASSWORD   "change-me-8chars"
#define OTA_PASSWORD         "change-me"          // asked for by the Arduino IDE when flashing over WiFi
