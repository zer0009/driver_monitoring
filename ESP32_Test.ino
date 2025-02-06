#include <SoftwareSerial.h>

// Pin definitions
#define GSM_RX 14
#define GSM_TX 12
#define GPS_RX 13
#define GPS_TX 15

// Input pins for Pi signals
#define SIGNAL_EYES_CLOSED 4
#define SIGNAL_YAWNING 5
#define SIGNAL_PHONE_USE 16

// Debug LED
#define DEBUG_LED 2

// Communication objects
SoftwareSerial gsmSerial(GSM_RX, GSM_TX);

// Test phone number
const String PHONE_NUMBER = "+201149920599";

void setup() {
    // Initialize Serial for debugging
    Serial.begin(115200);
    delay(2000);
    Serial.println("\n\n----- ESP32 Test Program Starting -----");
    
    // Setup pins
    pinMode(SIGNAL_EYES_CLOSED, INPUT);
    pinMode(SIGNAL_YAWNING, INPUT);
    pinMode(SIGNAL_PHONE_USE, INPUT);
    pinMode(DEBUG_LED, OUTPUT);
    
    // Initialize GSM
    Serial.println("\nInitializing GSM Serial...");
    gsmSerial.begin(9600);
    
    // Test all components
    testLED();
    testInputPins();
    testGSM();
}

void loop() {
    Serial.println("\n----- Testing Cycle Starting -----");
    
    // Test input pins
    testInputPins();
    
    // Test GSM
    testGSM();
    
    // Visual indicator
    blinkLED(1);
    
    // Wait before next test cycle
    delay(5000);
}

void testLED() {
    Serial.println("\nTesting Debug LED...");
    for(int i = 0; i < 3; i++) {
        digitalWrite(DEBUG_LED, LOW);  // LED ON
        Serial.println("LED ON");
        delay(500);
        digitalWrite(DEBUG_LED, HIGH); // LED OFF
        Serial.println("LED OFF");
        delay(500);
    }
}

void testInputPins() {
    Serial.println("\nTesting Input Pins...");
    
    // Test Eyes Closed Pin
    Serial.print("Eyes Closed Pin (");
    Serial.print(SIGNAL_EYES_CLOSED);
    Serial.print("): ");
    Serial.println(digitalRead(SIGNAL_EYES_CLOSED) ? "HIGH" : "LOW");
    
    // Test Yawning Pin
    Serial.print("Yawning Pin (");
    Serial.print(SIGNAL_YAWNING);
    Serial.print("): ");
    Serial.println(digitalRead(SIGNAL_YAWNING) ? "HIGH" : "LOW");
    
    // Test Phone Use Pin
    Serial.print("Phone Use Pin (");
    Serial.print(SIGNAL_PHONE_USE);
    Serial.print("): ");
    Serial.println(digitalRead(SIGNAL_PHONE_USE) ? "HIGH" : "LOW");
}

void testGSM() {
    Serial.println("\nTesting GSM Module...");
    
    // Basic AT command test
    Serial.println("Sending: AT");
    gsmSerial.println("AT");
    delay(1000);
    printGSMResponse();
    
    // Check if GSM is registered
    Serial.println("\nChecking network registration...");
    gsmSerial.println("AT+CREG?");
    delay(1000);
    printGSMResponse();
    
    // Check signal quality
    Serial.println("\nChecking signal quality...");
    gsmSerial.println("AT+CSQ");
    delay(1000);
    printGSMResponse();
    
    // Test SMS functionality
    sendTestSMS();
}

void sendTestSMS() {
    Serial.println("\nTesting SMS...");
    
    // Set SMS text mode
    gsmSerial.println("AT+CMGF=1");
    delay(1000);
    printGSMResponse();
    
    // Send test message
    Serial.println("Sending test SMS...");
    gsmSerial.println("AT+CMGS=\"" + PHONE_NUMBER + "\"");
    delay(1000);
    gsmSerial.print("ESP32 Test Message");
    delay(100);
    gsmSerial.write(26);  // CTRL+Z to send
    
    delay(5000);  // Wait for response
    printGSMResponse();
}

void printGSMResponse() {
    Serial.println("GSM Response:");
    while(gsmSerial.available()) {
        Serial.write(gsmSerial.read());
    }
    Serial.println();
}

void blinkLED(int times) {
    for(int i = 0; i < times; i++) {
        digitalWrite(DEBUG_LED, LOW);   // LED ON
        delay(200);
        digitalWrite(DEBUG_LED, HIGH);  // LED OFF
        delay(200);
    }
} 