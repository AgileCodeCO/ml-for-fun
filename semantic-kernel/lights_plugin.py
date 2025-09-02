from typing import Annotated
from semantic_kernel.functions import kernel_function

class LightsPlugin:
    lights = [
        {"id": 1, "name": "Table Lamp", "is_on": False},
        {"id": 2, "name": "Porch Light", "is_on": False},
        {"id": 3, "name": "Chandelier", "is_on": True},
    ]
    
    @kernel_function(
        name="get_lights",
        description="Get the list of lights and their current state"
    )
    
    def get_state(
        self
    ) -> str:
        """Get a list of all lights and their current state """
        return self.lights
    
    @kernel_function(
        name = "change_state",
        description = "Change the state of a light"
    )
    
    def change_state(
        self,
        id: int,
        is_on: bool
    ) -> str:
        """Change the state of a light"""
        for light in self.lights:
            if light["id"] == id:
                light["is_on"] = is_on
                return f"Light {light['name']} is now {'on' if is_on else 'off'}"
        return None