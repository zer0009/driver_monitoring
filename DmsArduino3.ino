#include <SoftwareSerial.h>
#include <TinyGPS++.h>

// Pin definitions
#define GSM_RX 14  // D5 on ESP8266
#define GSM_TX 12  // D6 on ESP8266
#define GPS_RX 13  // D7 on ESP8266
#define GPS_TX 15  // D8 on ESP8266

// Input pins for Pi signals
#define SIGNAL_EYES_CLOSED 4    // D2 on ESP8266
#define SIGNAL_YAWNING 5        // D1 on ESP8266
#define SIGNAL_PHONE_USE 16     // D0 on ESP8266

// Add debug LED pin
#define DEBUG_LED 2  // Built-in LED on ESP8266

// Communication objects
SoftwareSerial gsmSerial(GSM_RX, GSM_TX);
SoftwareSerial gpsSerial(GPS_RX, GPS_TX);
TinyGPSPlus gps;

// Phone number for alerts
const String PHONE_NUMBER = "+1234567890";  // Replace with your number

// Alert messages
const String ALERTS[] = {
    "Driver fatigue detected - Eyes closing",
    "Driver fatigue warning - Yawning",
    "Driver distraction - Phone usage"
};

// Add debounce time
const unsigned long DEBOUNCE_DELAY = 50;  // milliseconds

// Signal state tracking
struct Signal {
    const int pin;
    bool lastState;
    bool currentState;
    unsigned long lastAlertTime;
    unsigned long lastDebounceTime;  // Add debounce timing
    
    Signal(int p) : pin(p), lastState(false), currentState(false), 
                   lastAlertTime(0), lastDebounceTime(0) {}
};

Signal signals[] = {
    Signal(SIGNAL_EYES_CLOSED),
    Signal(SIGNAL_YAWNING),
    Signal(SIGNAL_PHONE_USE)
};

// Time between alerts for the same condition
const unsigned long ALERT_COOLDOWN = 300000;  // 5 minutes

void setup() {
    Serial.begin(9600);
    gsmSerial.begin(9600);
    gpsSerial.begin(9600);
    
    // Add debug LED setup
    pinMode(DEBUG_LED, OUTPUT);
    digitalWrite(DEBUG_LED, HIGH);  // LED off initially (ESP8266 LED is active LOW)
    
    // Configure input pins for receiving signals from Raspberry Pi
    for (Signal& signal : signals) {
        pinMode(signal.pin, INPUT);  // Simple INPUT mode is correct here
    }
    
    initGSM();
    
    // Visual feedback for initialization
    blinkLED(3);
    Serial.println("System initialized and monitoring...");
}

void loop() {
    // Update GPS data
    while (gpsSerial.available()) {
        gps.encode(gpsSerial.read());
    }
    
    // Check all input signals
    for (int i = 0; i < 3; i++) {
        checkSignal(i);
    }
}

void checkSignal(int signalIndex) {
    Signal& signal = signals[signalIndex];
    
    // Read digital value with debouncing
    bool reading = digitalRead(signal.pin);
    
    // If the state changed, reset debounce timer
    if (reading != signal.lastState) {
        signal.lastDebounceTime = millis();
    }
    
    // Check if debounce period has passed
    if ((millis() - signal.lastDebounceTime) > DEBOUNCE_DELAY) {
        // If state has changed and cooldown period has passed
        if (reading != signal.currentState) {
            signal.currentState = reading;
            
            if (reading && (millis() - signal.lastAlertTime > ALERT_COOLDOWN)) {
                signal.lastAlertTime = millis();
                digitalWrite(DEBUG_LED, LOW);  // LED on
                sendAlert(signalIndex);
                digitalWrite(DEBUG_LED, HIGH); // LED off
            }
        }
    }
    
    signal.lastState = reading;
}

// Add LED feedback function
void blinkLED(int times) {
    for(int i = 0; i < times; i++) {
        digitalWrite(DEBUG_LED, LOW);   // LED on
        delay(200);
        digitalWrite(DEBUG_LED, HIGH);  // LED off
        delay(200);
    }
}

void sendAlert(int alertIndex) {
    if (!gps.location.isValid()) {
        Serial.println("Warning: GPS location not available");
        return;
    }
    
    String message = createAlertMessage(ALERTS[alertIndex]);
    
    // Visual feedback before sending
    digitalWrite(DEBUG_LED, LOW);  // LED on
    
    // Send SMS with error checking
    gsmSerial.println("AT+CMGS=\"" + PHONE_NUMBER + "\"");
    delay(1000);
    gsmSerial.print(message);
    delay(100);
    gsmSerial.write(26);  // CTRL+Z to send
    
    Serial.println("Alert sent: " + message);
    
    // Visual feedback after sending
    digitalWrite(DEBUG_LED, HIGH);  // LED off
}

String createAlertMessage(const String& alert) {
    String message = "DRIVER SAFETY ALERT:\n" + alert + "\n\n";
    
    // Add location if available
    if (gps.location.isValid()) {
        String latitude = String(gps.location.lat(), 6);
        String longitude = String(gps.location.lng(), 6);
        message += "Location: " + latitude + "," + longitude + "\n";
        message += "Maps: http://maps.google.com/maps?q=" + latitude + "," + longitude + "\n";
    }
    
    // Add speed if available
    if (gps.speed.isValid()) {
        message += "Speed: " + String(gps.speed.kmph()) + " km/h\n";
    }
    
    // Add time if available
    if (gps.time.isValid()) {
        char timeStr[32];
        sprintf(timeStr, "Time: %02d:%02d:%02d UTC", 
                gps.time.hour(), gps.time.minute(), gps.time.second());
        message += timeStr;
    }
    
    return message;
}

void initGSM() {
    Serial.println("Initializing GSM...");
    delay(1000);
    
    gsmSerial.println("AT");
    delay(1000);
    
    gsmSerial.println("AT+CMGF=1"); // Set SMS text mode
    delay(1000);
    
    Serial.println("GSM initialized");
}