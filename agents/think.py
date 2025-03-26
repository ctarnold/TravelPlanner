# TO IMPLEMENT
# CONSIDERING REWRITING THINK, ACT CODE BASED ON TOOL_AGENTS
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools/planner")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../tools/planner")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from agents.local_model import LocalModel
from agents.prompts import THINK_PROMPT
from agents import tool_agents
from tool_agents import CityError, DateError


class Think:
    def __init__(self, model_path="/scratch/gpfs/ca2992/models/DeepSeek-R1-Distill-Llama-8B") -> None:
        self.llm = LocalModel(model_path=model_path)
        print("Think loaded.")
    def run(self, prompt: str, feedback: str) -> str:
        prompt = THINK_PROMPT + "\n Query: " + prompt + "\n\nFeedback: " + feedback
        return self.llm(prompt)
    
class Act:
    def __init__(self, model_path="/scratch/gpfs/ca2992/models/DeepSeek-R1-Distill-Llama-8B") -> None:
        self.llm = LocalModel(model_path=model_path)
        print("Act loaded.")
    def run(self, input: str) -> str:
        feedback = ""
        _, _, a = input.partition("BEGIN TOOL USAGE: ")
        for line in a.split(","):
            action_type, action_arg = tool_agents.parse_action(line)

            if action_type == 'FlightSearch':
                print("\n\nFlightSearch action reached\n", flush=True)
                try:
                    if tool_agents.validate_date_format(action_arg.split(', ')[2]) and tool_agents.validate_city_format(action_arg.split(', ')[0],self.city_set ) and tool_agents.validate_city_format(action_arg.split(', ')[1],self.city_set):
                        # self.scratchpad = self.scratchpad.replace(tool_agents.to_string(self.current_data).strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                        current_data = self.tools['flights'].run(action_arg.split(', ')[0], action_arg.split(', ')[1], action_arg.split(', ')[2])
                        current_observation = str(tool_agents.to_string(current_data))
                        feedback += current_observation +"\n"
                        # self.json_log[-1]['state'] = f'Successful'
                except Exception as e:
                    print(e)
                    # self.retry_record['flights'] += 1
                    current_observation = f'Illegal Flight Search. Please try again.'
                    feedback += f'Illegal Flight Search. Please try again.' + "\n"
                    # self.json_log[-1]['state'] = f'Illegal args. Other Error'
            elif action_type == 'AccommodationSearch':
                print("AccommodationSearch")
            elif action_type == 'AttractionSearch':
                print("AttractionSearch")
            elif action_type == 'RestaurantSearch':
                print("RestaurantSearch")
            elif action_type == 'NotebookWrite':
                print("NotebookWrite")
            elif action_type == 'Planner':
                print("Planner")        
            elif action_type == 'GoogleDistanceMatrix':
                print("GoogleDistanceMatrix")
            elif action_type == 'CitySearch':
                print("CitySearch")
            else:
                feedback += "Invalid tool usage: " + action_type + "\n"
        return feedback