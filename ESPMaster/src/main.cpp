#include <BluetoothSerial.h>
//https://github.com/hen1227/bluetooth-serial
#include <ArduinoJson.h>
#include <Arduino.h>

// Define RX and TX pins for Serial 2
#define RXD2 16
#define TXD2 17

#define RXD1 21
#define TXD1 19

#define ESP_BAUD 115200
#define GPS_BAUD 115200

// Create an instance of the HardwareSerial class for Serial 2
HardwareSerial gpsSerial(2);
HardwareSerial ESPSerial(1);

BluetoothSerial SerialBT;

const String slave1 = "ESP32_Slave_2.1";

struct msg_protocol
{
  int tipo;
  int ID;
  String istruzione;
  bool checkInvio;
};

char Json[255];

String ricezioneEV3(String& default_istruzione, bool& default_checkInvio) {
  int index = 0;
  
  while (gpsSerial.available() > 0) {
    char gpsData = gpsSerial.read();
    if (index < sizeof(Json) - 1) { // Ensure no buffer overflow
      Json[index++] = gpsData;
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
      //Serial.print("deserializeJson() failed: ");
      //Serial.println(error.f_str());
      memset(Json, 0, sizeof(Json));  
      return "";
    }

    msg_protocol msg_recvd;
    msg_recvd.tipo = msg_json["tipo"];
    msg_recvd.ID = msg_json["ID"];
    msg_recvd.checkInvio = msg_json["checkInvio"];
    msg_recvd.istruzione = msg_json["istruzione"].as<String>();

    memset(Json, 0, sizeof(Json)); // Clear the buffer before returning

    default_istruzione = msg_recvd.istruzione;
    default_checkInvio = msg_recvd.checkInvio;
  }
  
  return ""; // Return empty string if no data
}

String ricezioneRaspi(bool& checkInvio) {
  int index = 0;
  
  while (Serial.available() > 0) {
    char Data = Serial.read();
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
      //Serial.print("deserializeJson() failed: ");
      //Serial.println(error.f_str());
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
      //Serial.print("deserializeJson() failed: ");
      //Serial.println(error.f_str());
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

void invioEV3(int tipo_dispositivo, int ID_dispositivo, String istruzione) {
  StaticJsonDocument<255> msg_json;

      msg_json["tipo"] = tipo_dispositivo;
      msg_json["ID"] = ID_dispositivo;
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

  gpsSerial.write((const uint8_t*)Json, len);
}

void invioRaspi(int tipo, int ID, String istruzione) {
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

  Serial.write((const uint8_t*)Json, len);
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
  Serial.begin(115200);
  gpsSerial.begin(GPS_BAUD, SERIAL_8N1, RXD2, TXD2);
  ESPSerial.begin(ESP_BAUD, SERIAL_8N1, RXD1, TXD2);
  //Serial.println("Serial 2 started at 115200 baud rate");

  pinMode(BUILTIN_LED, OUTPUT);
  digitalWrite(BUILTIN_LED, LOW);

  // SerialBT.begin("ESPMaster", true);     OMESSO BLUETOOTH PER DEBUG
  // //Serial.println("ESPMaster ON");

  // while (!SerialBT.connect(slave1)){
  //   //Serial.println("Tentativo di connessione a 2.1...");
  //   delay(1000);
  // }
  // digitalWrite(BUILTIN_LED, HIGH);
  //Serial.println("Connesso");
}

String assegnaNome(int id) {
  return slave1;
}

void invioSicuroEsp(int id, String istruzione) {
  msg_protocol msg_recvd;
  const String slaveName = assegnaNome(id);
  // Serial.print("Connessione a: ");
  // Serial.println(slaveName);
  
  // while (!SerialBT.connect(slaveName)){
  //   Serial.println("Tentativo di connessione a 2.1...");
  //   delay(1000);
  // }

  if (SerialBT.connect(slaveName)) {
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

void invioSicuro(int tipo, int ID, String istruzione){

  if (tipo == 2)
  {
    if (ID == 1)
    {
      invioSicuroEsp(ID, istruzione);
    } else if (ID == 2 || ID == 3)
    {
      bool check;
      invioUARTesp(tipo, ID, istruzione);
      if (ricezioneUARTesp(check) == istruzione);
      {
        if (check)
        {
          digitalWrite(BUILTIN_LED, HIGH);
          delay(100);
          digitalWrite(BUILTIN_LED, LOW);
        }
        
      }
    }
  }
  else if (tipo == 4)
  {
    bool check;
    invioRaspi(tipo, ID, istruzione);
    if (ricezioneRaspi(check) == istruzione);
    {
      if (check)
      {
        digitalWrite(BUILTIN_LED, HIGH);
        delay(100);
        digitalWrite(BUILTIN_LED, LOW);
      }
      
    }        
  }
  else
  {
    unsigned long inizio = millis();

    invioEV3(tipo, ID, istruzione);

    String ist = "";
    bool checkInvio = false;
    while (checkInvio == false && ist != istruzione){
      if (tipo == 1 || tipo == 3){
        ricezioneEV3(ist, checkInvio);
      }

      if (millis() - inizio > 15000){
        //Serial.println("###ERRORE 001 TIMEOUT###");
        break;
      }
      if (ist != istruzione && checkInvio == true){
        //Serial.print("###ERRORE 003 BUSYMASTER###");
      }
      if (ist == istruzione && checkInvio == false){
        //Serial.print("###ERRORE 004 ###");
      }
    }
    if (checkInvio == true && ist == istruzione)
    {
      //Serial.println("###MESSAGGIO RICEVUTO CON SUCCESSO###");
    }
  }
}

String msgConferma = " ";
String immagine = " ";

void loop() {
  if (Serial.available())
  {
    do
    {
      bool check;
      immagine = ricezioneRaspi(check);
      invioRaspi(4, 0, immagine);
      msgConferma = ricezioneRaspi(check);
    } while (msgConferma == "ERR");

    if (immagine == "-cover-")
    {
      //
    } else if (immagine == "-rabbit-")
    {
      invioSicuro(3, 2, "scenaConiglio");
    } else if (immagine == "-cat-")
    {
      invioSicuro(2, 2, "scenaGatto1");
    } else if (immagine == "-caterpillar-")
    {
      invioSicuro(3, 3, "scenaBruco");
    } else if (immagine == "-queen-")
    {
      invioSicuro(3, 1, "avvioGonna");
      invioSicuro(2, 1, "incazzati");
    } else if (immagine == "-hatman-")
    {
      invioSicuro(3, 1, "arrestoGonna");
      invioSicuro(2, 1, "calmati");
      invioSicuro(2, 2, "scenagatto2");
    } else if (immagine == "-alice-")
    {
      invioSicuro(3, 2, "scenaConiglioFinale");
      invioSicuro(3, 1, "avvioGonnaFinale");
      invioSicuro(2, 1, "incazzatiFinale");
      invioSicuro(2, 2, "scenagattoFinale");

    }
  }
}

