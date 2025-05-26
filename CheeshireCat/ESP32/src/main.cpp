#include <BluetoothSerial.h>
//https://github.com/hen1227/bluetooth-serial
#include <ArduinoJson.h>
#include <Arduino.h>
#include <ESP32Servo.h>

BluetoothSerial SerialBT;
Servo coda;
bool statusCoda = false;   // Se false coda ferma, se true coda in movimento

const int M1orario = 12;
const int M2orario = 27;
const int M1antiorario = 14;
const int M2antiorario = 26;

struct msg_protocol
{
  int tipo;
  int ID;
  String istruzione;
  bool checkInvio;
};

char Json[255];

void spegnitutto() {
  digitalWrite(M1orario, LOW);
  digitalWrite(M2orario, LOW);
  digitalWrite(M1antiorario, LOW);
  digitalWrite(M2antiorario, LOW);
}

void setup() {
  Serial.begin(115200);
  coda.attach(13);
  coda.write(70);
  pinMode(M1orario, OUTPUT);
  pinMode(M2orario, OUTPUT);
  pinMode(M1antiorario, OUTPUT);
  pinMode(M2antiorario, OUTPUT);
  pinMode(33, OUTPUT);
  pinMode(23, OUTPUT);
  pinMode(BUILTIN_LED, OUTPUT);
  digitalWrite(BUILTIN_LED, LOW);

  SerialBT.begin("ESP32_Slave_2.2");
  if (Serial) Serial.println("ESP32 Slave 2.2 Bluetooth ON, in attesa di connessione...");
  digitalWrite(BUILTIN_LED, HIGH);
  spegnitutto();
}

void sano(bool sens) {
  if (sens) {
    digitalWrite(M1orario, HIGH);
    digitalWrite(M2orario, HIGH);
  } else {
    digitalWrite(M1antiorario, HIGH);
    digitalWrite(M2antiorario, HIGH);
  }
}

void strabico(bool sens) {
  if (sens) {
    digitalWrite(M1orario, HIGH);
    digitalWrite(M2antiorario, HIGH);
  } else {
    digitalWrite(M1antiorario, HIGH);
    digitalWrite(M2orario, HIGH);
  }
}

void scenaGatto(int n)
{
  digitalWrite(23, HIGH);
  delay(50);
  digitalWrite(23, LOW);
  for (int i = 0; i < n; i++)
  {
    coda.write(70);
    sano(true);
    delay(2000);
    spegnitutto();
    coda.write(40);
    coda.write(10);
    sano(false);
    delay(2000);
    spegnitutto();
    coda.write(40);
    coda.write(70);
    strabico(true);
    delay(2000);
    spegnitutto();
    coda.write(40);
    coda.write(10);
    strabico(false);
    spegnitutto();
  }
  digitalWrite(23, HIGH);
  delay(50);
  digitalWrite(23, LOW);
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

    if (msg_recvd.istruzione == "scenaGatto1") {
      scenaGatto(4);
    }
    else if (msg_recvd.istruzione == "scenaGatto2") {
      scenaGatto(1);
    }
    else if (msg_recvd.istruzione == "scenaGattoFinale")
    {
      scenaGatto(2);
    }
  }  
}

