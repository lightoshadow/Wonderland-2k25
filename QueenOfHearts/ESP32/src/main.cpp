#include <Arduino.h>
#include <ESP32Servo.h>
#include <ArduinoJson.h>
#include <BluetoothSerial.h>

// Oggetto BluetoothSerial per la comunicazione
BluetoothSerial SerialBT;

static const int PINbraccioSX = 32;
static const int PINbraccioDX = 23;
Servo braccioSX;
Servo braccioDX;

bool status = false;   // Se false occhi RGB e braccia fermae, se true occhi rossi e braccia in movimento

const int Rosso1 = 26;
const int Verde1 = 25;
const int Blu1 = 33;
const int Rosso2 = 22;
const int Verde2 = 19;
const int Blu2 = 18;

const int statusLED = 14;

struct msg_protocol {
  int tipo;
  int ID;
  String istruzione;
  bool checkInvio;
};

void setup() {
  Serial.begin(115200);

  // Timeout per la seriale: attende fino a 3 secondi ma non blocca l'esecuzione
  unsigned long start = millis();
  while (!Serial && millis() - start < 3000) {}

  pinMode(BUILTIN_LED, OUTPUT);
  digitalWrite(BUILTIN_LED, LOW);

  braccioSX.attach(PINbraccioSX);
  braccioDX.attach(PINbraccioDX);

  pinMode(Rosso1, OUTPUT);
  pinMode(Verde1, OUTPUT);
  pinMode(Blu1, OUTPUT);
  pinMode(Rosso2, OUTPUT);
  pinMode(Verde2, OUTPUT);
  pinMode(Blu2, OUTPUT);

  pinMode(statusLED, OUTPUT);
  digitalWrite(statusLED, LOW);

  SerialBT.begin("ESP32_Slave_2.1");
  if (Serial) Serial.println("ESP32 Slave 2.1 Bluetooth ON, in attesa di connessione...");
  digitalWrite(statusLED, HIGH);
}

void setColor(int redValue, int greenValue,  int blueValue) {
  analogWrite(Rosso1, redValue);
  analogWrite(Rosso2, redValue);
  analogWrite(Verde1,  greenValue);
  analogWrite(Verde2, greenValue);
  analogWrite(Blu1, blueValue);
  analogWrite(Blu2, blueValue);
}

void calma() {
  braccioSX.write(20);
  braccioDX.write(180);
  setColor(255, 255, 255); // White Color
  delay(2000);
  setColor(0,0,0);
  delay(40);
}

void Incazzata() {
  setColor(255, 0, 0);
  braccioSX.write(180);
  delay(1000);
  braccioSX.write(20);
  braccioDX.write(20);
  delay(1000);
  braccioDX.write(180);
}

void loop() {
  if (SerialBT.connected()) {
    Serial.println("Connesso.");
    digitalWrite(BUILTIN_LED, HIGH);
  } else {
    Serial.println("In attesa di connessione...");
  }
  if (SerialBT.available()) {
    digitalWrite(BUILTIN_LED, LOW);
    delay(50);
    digitalWrite(BUILTIN_LED, HIGH);
    String Json = SerialBT.readString();
    
    if (Serial) {
      Serial.print("Ricevuto JSON: ");
      Serial.println(Json);
    }

    if (Json.endsWith("!")) {
      Json.remove(Json.length() - 1);
    }

    StaticJsonDocument<255> msg_json;
    DeserializationError error = deserializeJson(msg_json, Json);

    if (error) {
      if (Serial) {
        Serial.print("Errore deserializzazione JSON: ");
        Serial.println(error.c_str());
      }
      return;
    }

    msg_protocol msg_recvd;
    msg_recvd.tipo = msg_json["tipo"];
    msg_recvd.ID = msg_json["ID"];
    msg_recvd.checkInvio = msg_json["checkInvio"];
    msg_recvd.istruzione = msg_json["istruzione"].as<String>();

    msg_recvd.checkInvio = true;
    msg_json["checkInvio"] = msg_recvd.checkInvio;

    char JsonOut[255];
    size_t len = serializeJson(msg_json, JsonOut);

    if (len < sizeof(JsonOut) - 1) {
      JsonOut[len] = '!';
      JsonOut[len + 1] = '\0';
      len++;
    }

    SerialBT.write((const uint8_t*)JsonOut, len);

    if (Serial) {
      Serial.println("Messaggio inviato via Bluetooth");
    }

    if (msg_recvd.istruzione == "incazzati") {
      if (Serial) Serial.println("Comando ricevuto: avvio");
      status = true; 
    }
    if (msg_recvd.istruzione == "calmati") {
      if (Serial) Serial.println("Comando ricevuto: pausa");
      status = false; 
    }
  }

  if (!status) {
    calma();
  } else {
    Incazzata();
  }

  delay(50);
}