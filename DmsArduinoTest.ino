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

// Phone number for alerts - Egyptian format
const String PHONE_NUMBER = "+201067872456";  // Replace X with your actual number
// Example: if your number is 012-3456-7890, use "+2012XXXXXXXX"

// Alert messages
const String ALERTS[] = {
    "Driver fatigue detected - Eyes closing",
    "Driver fatigue warning - Yawning",
    "Driver distraction - Phone usage"
};

// Add debounce time
const unsigned long DEBOUNCE_DELAY = 50;  // milliseconds

// Add test mode flag
bool TEST_MODE = true;

// Add test button pin
#define TEST_BUTTON_PIN 0  // GPIO0, typically FLASH button on ESP8266

// Modify Signal struct to include simulation
struct Signal {
    const int pin;
    bool lastState;
    bool currentState;
    unsigned long lastAlertTime;
    unsigned long lastDebounceTime;
    bool simulatedState;  // New field for test mode
    
    Signal(int p) : pin(p), lastState(false), currentState(false), 
                   lastAlertTime(0), lastDebounceTime(0), simulatedState(false) {}
};

Signal signals[] = {
    Signal(SIGNAL_EYES_CLOSED),
    Signal(SIGNAL_YAWNING),
    Signal(SIGNAL_PHONE_USE)
};

// Time between alerts for the same condition
const unsigned long ALERT_COOLDOWN = 300000;  // 5 minutes

// Test message
const String TEST_MESSAGE = "This is a test message from DMS Arduino";

void setup() {
    // Start Serial first for debugging
    Serial.begin(115200);
    delay(2000);  // Wait for Serial to be ready
    
    Serial.println("\n\n----- DMS Arduino Starting -----");
    
    if (TEST_MODE) {
        pinMode(TEST_BUTTON_PIN, INPUT_PULLUP);
        Serial.println("Running in TEST MODE");
        Serial.println("Press FLASH button to trigger test sequence");
    } else {
        // Original pin setup
        pinMode(SIGNAL_EYES_CLOSED, INPUT);
        pinMode(SIGNAL_YAWNING, INPUT);
        pinMode(SIGNAL_PHONE_USE, INPUT);
    }
    
    // Print initial pin states
    Serial.println("\nInitial Pin States:");
    Serial.print("Eyes Closed Pin (");
    Serial.print(SIGNAL_EYES_CLOSED);
    Serial.print("): ");
    Serial.println(digitalRead(SIGNAL_EYES_CLOSED));
    
    Serial.print("Yawning Pin (");
    Serial.print(SIGNAL_YAWNING);
    Serial.print("): ");
    Serial.println(digitalRead(SIGNAL_YAWNING));
    
    Serial.print("Phone Use Pin (");
    Serial.print(SIGNAL_PHONE_USE);
    Serial.print("): ");
    Serial.println(digitalRead(SIGNAL_PHONE_USE));
    
    // Setup LED
    pinMode(DEBUG_LED, OUTPUT);
    digitalWrite(DEBUG_LED, HIGH);  // LED off initially
    
    // Visual indicator that code is running
    blinkLED(5);  // Blink 5 times at startup
    
    Serial.println("Starting GSM Serial...");
    gsmSerial.begin(9600);
    
    // Basic AT command test
    Serial.println("\nTesting GSM Module with AT command:");
    testGSM();
    
    // Initialize GSM in setup
    initGSM();
}

void loop() {
    if (TEST_MODE) {
        handleTestMode();
    } else {
        // Original signal checking code
        for(int i = 0; i < 3; i++) {
            checkSignal(i);
        }
    }
    delay(1000);
}

void testGSM() {
    digitalWrite(DEBUG_LED, LOW);  // LED on while testing
    
    Serial.println("Sending: AT");
    gsmSerial.println("AT");
    delay(1000);
    
    if (gsmSerial.available()) {
        Serial.println("Response received:");
        while (gsmSerial.available()) {
            char c = gsmSerial.read();
            Serial.write(c);
        }
    } else {
        Serial.println("NO RESPONSE FROM GSM MODULE!");
    }
    
    digitalWrite(DEBUG_LED, HIGH);  // LED off after test
}

void blinkLED(int times) {
    Serial.print("Blinking LED ");
    Serial.print(times);
    Serial.println(" times");
    
    for(int i = 0; i < times; i++) {
        digitalWrite(DEBUG_LED, LOW);   // LED on
        delay(200);
        digitalWrite(DEBUG_LED, HIGH);  // LED off
        delay(200);
    }
}

void checkSignal(int signalIndex) {
    Signal& signal = signals[signalIndex];
    
    // Read digital value with debouncing
    bool reading = digitalRead(signal.pin);
    
    // Print current reading
    Serial.print("Signal ");
    Serial.print(signalIndex);
    Serial.print(" (Pin ");
    Serial.print(signal.pin);
    Serial.print("): ");
    Serial.println(reading ? "HIGH" : "LOW");
    
    // If the state changed, reset debounce timer
    if (reading != signal.lastState) {
        signal.lastDebounceTime = millis();
        Serial.println("State changed - debouncing...");
    }
    
    // Check if debounce period has passed
    if ((millis() - signal.lastDebounceTime) > DEBOUNCE_DELAY) {
        // If state has changed and cooldown period has passed
        if (reading != signal.currentState) {
            signal.currentState = reading;
            
            if (reading && (millis() - signal.lastAlertTime > ALERT_COOLDOWN)) {
                signal.lastAlertTime = millis();
                Serial.print("ALERT TRIGGERED for signal ");
                Serial.println(signalIndex);
                sendAlert(signalIndex);
            }
        }
    }
    
    signal.lastState = reading;
}

void sendAlert(int alertIndex) {
    Serial.print("\nPreparing to send alert: ");
    Serial.println(ALERTS[alertIndex]);
    
    if (!gps.location.isValid()) {
        Serial.println("Warning: GPS location not available");
        return;
    }
    
    String message = createAlertMessage(ALERTS[alertIndex]);
    Serial.println("Message content:");
    Serial.println(message);
    
    // Visual feedback before sending
    digitalWrite(DEBUG_LED, LOW);  // LED on
    
    Serial.println("Sending SMS...");
    // Send SMS with error checking
    gsmSerial.println("AT+CMGS=\"" + PHONE_NUMBER + "\"");
    delay(1000);
    gsmSerial.print(message);
    delay(100);
    gsmSerial.write(26);  // CTRL+Z to send
    
    Serial.println("SMS command sent, waiting for response...");
    delay(5000);  // Wait for GSM response
    printGSMResponse();  // Print any response from the GSM module
    
    Serial.println("Alert process completed");
    
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
    delay(2000);
    
    Serial.println("\nTesting AT");
    gsmSerial.println("AT");
    delay(1000);
    printGSMResponse();
    
    Serial.println("\nSetting SMS mode");
    gsmSerial.println("AT+CMGF=1");
    delay(1000);
    printGSMResponse();
    
    Serial.println("\nSetting character encoding");
    gsmSerial.println("AT+CSCS=\"GSM\"");
    delay(1000);
    printGSMResponse();
    
    Serial.println("\nChecking network registration");
    gsmSerial.println("AT+CREG?");
    delay(1000);
    printGSMResponse();
    
    Serial.println("\nChecking signal quality");
    gsmSerial.println("AT+CSQ");
    delay(1000);
    printGSMResponse();
    
    Serial.println("GSM initialization complete");
}

void printGSMResponse() {
    while (gsmSerial.available()) {
        Serial.write(gsmSerial.read());
    }
    Serial.println();
}

void sendTestMessage() {
    digitalWrite(DEBUG_LED, LOW);  // LED on
    
    Serial.println("Sending to number: " + PHONE_NUMBER);
    gsmSerial.println("AT+CMGS=\"" + PHONE_NUMBER + "\"");
    delay(1000);
    printGSMResponse();
    
    gsmSerial.print(TEST_MESSAGE);
    Serial.println("Message content: " + TEST_MESSAGE);
    delay(100);
    
    gsmSerial.write(26);  // CTRL+Z to send
    delay(1000);
    printGSMResponse();
    
    digitalWrite(DEBUG_LED, HIGH);  // LED off
}

void handleTestMode() {
    static unsigned long lastTestTime = 0;
    static int testPhase = 0;
    
    // Check if test button is pressed (FLASH button is active LOW)
    if (digitalRead(TEST_BUTTON_PIN) == LOW) {
        if (millis() - lastTestTime > 5000) {  // Prevent rapid retriggering
            lastTestTime = millis();
            
            Serial.println("\n----- Starting Test Sequence -----");
            
            // Simulate each alert type with a delay between them
            switch(testPhase) {
                case 0:
                    Serial.println("Testing Eyes Closed Alert");
                    sendAlert(0);
                    break;
                case 1:
                    Serial.println("Testing Yawning Alert");
                    sendAlert(1);
                    break;
                case 2:
                    Serial.println("Testing Phone Usage Alert");
                    sendAlert(2);
                    break;
            }
            
            testPhase = (testPhase + 1) % 3;  // Cycle through test phases
            blinkLED(2);  // Visual feedback
        }
    }
    
    // Add fake GPS data for testing
    while (gpsSerial.available()) {
        gps.encode(gpsSerial.read());
    }
    
    // Simulate GPS data if not available
    if (!gps.location.isValid()) {
        // Simulate location near Cairo, Egypt
        gps.location.setLatitude(30.0444);
        gps.location.setLongitude(31.2357);
        gps.speed.setSpeedKmph(60.5);
        gps.time.setTime(12, 30, 0);  // Set time to 12:30:00
    }
}