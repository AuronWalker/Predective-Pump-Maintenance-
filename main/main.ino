#include <Arduino.h>
#include "model.h"

Eloquent::ML::Port::RandomForest model;

void setup() {
  Serial.begin(115200);

  float features[] = {0.05,130,22.27,0.066,127.81,21.78,0.00,18.89,0.08,0.00};
  
  unsigned long start = micros();
  Serial.println(start);
  int y = model.predict(features);
  unsigned long end = micros();
  Serial.println(end);
  
  Serial.print("Inference time (µs): ");
  Serial.println(end - start);
}


void loop() {

}