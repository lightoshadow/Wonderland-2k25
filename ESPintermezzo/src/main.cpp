#include <Arduino.h>
#include <BluetoothSerial.h>
//https://github.com/hen1227/bluetooth-serial
#include <ArduinoJson.h>
#include <Arduino.h>

#define RXD1 21
#define TXD1 19

#define ESP_BAUD 115200
#define GPS_BAUD 115200

HardwareSerial ESPSerial(1);
BluetoothSerial SerialBT;

char Json[255];

const String slave2 = "ESP_Slave_2.2"; // hostname of the slave device

struct msg_protocol
{
  int tipo;
  int ID;
  String istruzione;
  bool checkInvio;
};

String ricezioneUARTesp(bool& checkInvio) {
  int index = 0;
  
  while (ESPSerial.available() > 0) {
    char Data = ESPSerial.read();
    if (index < sizeof(Json) - 1) { // Ensure no buffer overflow
      Json[index++] = Data;
      //Serial.print(gpsData);
    }
  }
  
  // Controlla se l'ultimo carattere è "!" e lo rimuove
  if (index > 0 && Json[index - 1] == '!') {
    index--;  // Riduce la lunghezza effettiva del JSON ricevuto
  }

  Json[index] = '\0'; // Ensure it's a null-terminated string
  
  if (index > 0) {
    StaticJsonDocument<255> msg_json; // Proper JSON buffer
    DeserializationError error = deserializeJson(msg_json, Json);

    if (error) {
      Serial.print("deserializeJson() failed: ");
      Serial.println(error.f_str());
      memset(Json, 0, sizeof(Json));  
      return "";
    }

    msg_protocol msg_recvd;
    msg_recvd.tipo = msg_json["tipo"];
    msg_recvd.ID = msg_json["ID"];
    msg_recvd.checkInvio = msg_json["checkInvio"];
    msg_recvd.istruzione = msg_json["istruzione"].as<String>();

    memset(Json, 0, sizeof(Json)); // Clear the buffer before returning

    checkInvio = msg_recvd.checkInvio;
    return msg_recvd.istruzione;
  }
  
  return ""; // Return empty string if no data
}

void invioUARTesp(int tipo, int ID, String istruzione) {
  StaticJsonDocument<255> msg_json;

      msg_json["tipo"] = tipo;
      msg_json["ID"] = ID;
      msg_json["istruzione"] = istruzione;
      msg_json["checkInvio"] = false;

      memset(Json, 0, sizeof(Json));

      // Serializza il JSON in una stringa
      size_t len = serializeJson(msg_json, Json, sizeof(Json));

      // Aggiungi "!" alla fine del messaggio
      if (len < sizeof(Json)) {  // Assicurati che ci sia spazio per il carattere "!"
          Json[len] = '!';      // Aggiungi "!" alla fine
          len++;                // Incrementa la lunghezza del messaggio
      }

  ESPSerial.write((const uint8_t*)Json, len);
}


void setup() {
  // put your setup code here, to run once:
  ESPSerial.begin(115200, SERIAL_8N1, RXD1, TXD1);
  Serial.begin(115200);

  SerialBT.begin("ESP32_Slave_2.3", true);    
  Serial.println("ESPIntermezzo ON");

  while (!SerialBT.connect(slave2)) {
    //Serial.println("Tentativo di connessione a 2.1...");
    delay(100);
  }
  digitalWrite(BUILTIN_LED, HIGH);
  // Serial.println("Connesso");
}

void invioSicuroEsp(int id, String istruzione) {
  msg_protocol msg_recvd;
  const String slaveName = slave2;
  // Serial.print("Connessione a: ");
  // Serial.println(slaveName);
  
  // while (!SerialBT.connect(slaveName)){
  //   Serial.println("Tentativo di connessione a 2.1...");
  //   delay(1000);
  // }

  if (SerialBT.connect(slave2)) {
      StaticJsonDocument<255> msg_json;

      msg_json["tipo"] = 2;
      msg_json["ID"] = id;
      msg_json["istruzione"] = istruzione;
      msg_json["checkInvio"] = false;

      memset(Json, 0, sizeof(Json));

      // Serializza il JSON in una stringa
      size_t len = serializeJson(msg_json, Json, sizeof(Json));

      // Aggiungi "!" alla fine del messaggio
      if (len < sizeof(Json)) {  // Assicurati che ci sia spazio per il carattere "!"
          Json[len] = '!';      // Aggiungi "!" alla fine
          len++;                // Incrementa la lunghezza del messaggio
      }


      //Serial.println("Connesso con successo!");
      digitalWrite(BUILTIN_LED, HIGH);
      SerialBT.write((const uint8_t*)Json, len);
      
      // Attendi la risposta dallo Slave
      unsigned long startTime = millis();
      bool ricvd = false;
      while (millis() - startTime < 15000 && ricvd == false) {
          if (SerialBT.available()) {
            String messaggioJson = SerialBT.readStringUntil('\n');
            StaticJsonDocument<255> msg_json; // Proper JSON buffer
            DeserializationError error = deserializeJson(msg_json, messaggioJson);
        
            if (error) {
              //Serial.print("deserializeJson() failed: ");
              //Serial.println(error.f_str());
            }
        
            // Usa il puntatore sg_recvd per aggiornare la struttura passata per riferimento
            msg_recvd.tipo = msg_json["tipo"];
            msg_recvd.ID = msg_json["ID"];
            msg_recvd.checkInvio = msg_json["checkInvio"];
            msg_recvd.istruzione = msg_json["istruzione"].as<String>();

            if (msg_recvd.istruzione != istruzione && msg_recvd.checkInvio == true){
              //Serial.print("###ERRORE 003 BUSYMASTER###");
            }
            if (msg_recvd.istruzione == istruzione && msg_recvd.checkInvio == false){
              //Serial.print("###ERRORE 004 ###");
            }

            ricvd = true;
            break;
          }
      }
      if (!ricvd)
      {
        //Serial.println("TIMEOUT");
      }
      if (msg_recvd.istruzione == istruzione && msg_recvd.checkInvio == true)
      {
        //Serial.println("MESSAGGIO RICEVUTO CON SUCCESSO");
      }
      

      // Serial.println("Disconnessione...");
      // digitalWrite(BUILTIN_LED, LOW);
      // SerialBT.disconnect();
      // delay(2000); // Pausa prima della prossima connessione
  } else {
      //Serial.println("Connessione fallita!");
  }
}

String messaggio;
bool check;

void loop() {
  messaggio = ricezioneUARTesp(check);
  invioSicuroEsp(3, messaggio);
  invioUARTesp(0, 0, messaggio);
}