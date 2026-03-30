import paho.mqtt.client as mqtt
import requests
import os
import logging
import json

class IoTHubModule:
    def __init__(self):
        self.ha_url = os.getenv("HOME_ASSISTANT_URL")
        self.ha_token = os.getenv("HOME_ASSISTANT_TOKEN")
        self.mqtt_host = os.getenv("MQTT_HOST", "localhost")
        self.mqtt_port = int(os.getenv("MQTT_PORT", 1883))
        self.logger = logging.getLogger("IoTHubModule")

    def control_home_assistant(self, entity_id, service="toggle", domain="light"):
        """Controls an entity in Home Assistant."""
        if not self.ha_url or not self.ha_token:
            return False, "Home Assistant credentials missing."
        
        url = f"{self.ha_url}/api/services/{domain}/{service}"
        headers = {
            "Authorization": f"Bearer {self.ha_token}",
            "content-type": "application/json",
        }
        data = {"entity_id": entity_id}
        
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                return True, f"Successfully executed {service} on {entity_id}."
            else:
                return False, f"Failed with status code {response.status_code}: {response.text}"
        except Exception as e:
            self.logger.error(f"Home Assistant error: {e}")
            return False, f"Error: {e}"

    def publish_mqtt(self, topic, message):
        """Publishes a message to an MQTT broker."""
        try:
            client = mqtt.Client()
            client.connect(self.mqtt_host, self.mqtt_port, 60)
            client.publish(topic, message)
            client.disconnect()
            return True, f"Published to {topic}."
        except Exception as e:
            self.logger.error(f"MQTT error: {e}")
            return False, f"Error: {e}"

    def run_scene(self, scene_name):
        """Specifically runs a Home Assistant scene."""
        return self.control_home_assistant(f"scene.{scene_name}", service="turn_on", domain="scene")
