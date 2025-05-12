#include <BluetoothSerial.h>
//https://github.com/hen1227/bluetooth-serial
#include <ArduinoJson.h>
#include <Arduino.h>
#include <ESP32Servo.h>


// Define RX and TX pins for Serial 2
#define RXD2 16
#define TXD2 17

#define GPS_BAUD 115200

// Create an instance of the HardwareSerial class for Serial 2
HardwareSerial gpsSerial(2);

BluetoothSerial SerialBT;
Servo coda;
bool statusCoda = false;   // Se false coda ferma, se true coda in movimento

const int M1orario = 12;
const int M2orario = 27;
const int M1antiorario = 14;
const int M2antiorario = 26;

const String slave1 = "ESP32_Slave_2.1";
const String slave2 = "ESP32_Slave_2.2";
const String slave3 = "ESP32_Slave_2.3";

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
      Serial.print(gpsData);
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

    default_istruzione = msg_recvd.istruzione;
    default_checkInvio = msg_recvd.checkInvio;
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

void spegnitutto() {
  digitalWrite(M1orario, LOW);
  digitalWrite(M2orario, LOW);
  digitalWrite(M1antiorario, LOW);
  digitalWrite(M2antiorario, LOW);
}

void setup() {
  Serial.begin(115200);
  gpsSerial.begin(GPS_BAUD, SERIAL_8N1, RXD2, TXD2);
  Serial.println("Serial 2 started at 115200 baud rate");

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

  SerialBT.begin("ESPMaster", true);
  Serial.println("ESPMaster ON");

  while (!SerialBT.connect(slave1)){
    Serial.println("Tentativo di connessione a 2.1...");
    delay(1000);
  }
  digitalWrite(BUILTIN_LED, HIGH);
  Serial.println("Connesso");
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

String assegnaNome(int id) {
  if (id == 1)
  {
    return slave1;
  } else if (id == 2)
  {
    return slave2;
  } else if (id == 3)
  {
    return slave3;
  } else
  {
    Serial.println("ERRORE id sbagliato");
  }  
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


      Serial.println("Connesso con successo!");
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
              Serial.print("deserializeJson() failed: ");
              Serial.println(error.f_str());
            }
        
            // Usa il puntatore sg_recvd per aggiornare la struttura passata per riferimento
            msg_recvd.tipo = msg_json["tipo"];
            msg_recvd.ID = msg_json["ID"];
            msg_recvd.checkInvio = msg_json["checkInvio"];
            msg_recvd.istruzione = msg_json["istruzione"].as<String>();

            if (msg_recvd.istruzione != istruzione && msg_recvd.checkInvio == true){
              Serial.print("###ERRORE 003 BUSYMASTER###");
            }
            if (msg_recvd.istruzione == istruzione && msg_recvd.checkInvio == false){
              Serial.print("###ERRORE 004 ###");
            }

            ricvd = true;
            break;
          }
      }
      if (!ricvd)
      {
        Serial.println("TIMEOUT");
      }
      if (msg_recvd.istruzione == istruzione && msg_recvd.checkInvio == true)
      {
        Serial.println("MESSAGGIO RICEVUTO CON SUCCESSO");
      }
      

      // Serial.println("Disconnessione...");
      // digitalWrite(BUILTIN_LED, LOW);
      // SerialBT.disconnect();
      // delay(2000); // Pausa prima della prossima connessione
  } else {
      Serial.println("Connessione fallita!");
  }
}

void invioSicuro(int tipo, int ID, String istruzione){

  if (tipo == 2)
  {
    invioSicuroEsp(ID, istruzione);
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
        Serial.println("###ERRORE 001 TIMEOUT###");
        break;
      }
      if (ist != istruzione && checkInvio == true){
        Serial.print("###ERRORE 003 BUSYMASTER###");
      }
      if (ist == istruzione && checkInvio == false){
        Serial.print("###ERRORE 004 ###");
      }
    }
    if (checkInvio == true && ist == istruzione)
    {
      Serial.println("###MESSAGGIO RICEVUTO CON SUCCESSO###");
    }
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
  //lista comandi 
  digitalWrite(33, LOW);
    bool checkInvio = false;
    String ist = "";
    while (checkInvio == false && ist == ""){
        ricezioneEV3(ist, checkInvio);
    }
    Serial.println("Pulsante premuto");
    digitalWrite(33, HIGH);
    spegnitutto();
    delay(6000);
    scenaGatto(4);
    delay(18000);
    invioSicuro(3,1,"avvioGonna");
    invioSicuro(2,1,"incazzati");    
    delay(7000);
    invioSicuro(3,1,"arrestoGonna");
    invioSicuro(2,1,"calmati");
    delay(3000);
    scenaGatto(1);
    digitalWrite(BUILTIN_LED, LOW);

  delay(2000);
  Serial.println("-------------------------------");
}
