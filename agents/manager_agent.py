from typing import List, Dict, Any
from .tool_agents import ReactAgent
from .prompts import manager_agent_prompt  # You might need to adjust this import

class ManagerAgent:
    def __init__(self,
                 query: str,
                 tools: List[str],
                 react_llm_name: str = 'local',
                 planner_llm_name: str = 'local',
                 max_iterations: int = 1,
                 evaluation_criteria: List[str] = None):
        """
        Initializes the ManagerAgent.
        """
        self.query = query
        self.react_agent = ReactAgent(
            args=None,
            tools=tools,
            react_llm_name=react_llm_name,
            planner_llm_name=planner_llm_name,
            max_steps=10
        )
        print("\nPlanner LLM: ", planner_llm_name)
        print("\nReact LLM: ", react_llm_name)
        self.max_iterations = max_iterations
        self.evaluation_criteria = evaluation_criteria or [
            "Budget Compliance",
            "Room Rules",
            "Room Type",
            "Cuisine",
            "Transportation"
            "Common Sense"
        ]

    def evaluate_plan(self, plan: str) -> Dict[str, str]:  # Changed return type to Dict[str, str]
        """
        Evaluates the plan against the criteria and returns feedback.
        """
        # For now have fixed feedback
        feedback = {}
        feedback["Budget Compliance"] = "Total the costs of the plan you made and compare it to the user's budget. Be strictly below budget."
        feedback["Room Rules"] = "Next, verify that the accommodations meet all room rules. Does the user want a no smoking room, a no parties room? Does the user have pets? Look for these restrictions and all other preferences in the request. Call the accommodations tool."
        feedback["Room Type"] = "Check out the room type in the original request and the room you selected. Does it fit the number of people? Does the user want an Entire Room, a Private Room, etc? Call the accommodations tool."
        feedback["Cuisine"] = "If the user has a dietary restriction, check every single restaurant that it is compatible with this dietary restriction. Check that e restaurants are of the desired types, Chinese, American, Italian, etc. Call the restaurants tool."
        feedback["Transportation"] = "Ensure that transporation is in line with user needs. Does the user want to drive, fly, etc? Call the flights tool. Call the distance tool."
        feedback["Common Sense"] = "Verify the plan for other reasoning constraints. Is everything physically possible? Does the plan make sense?"
        return feedback

    def refine_plan(self, plan: str, feedback: Dict[str, str]) -> str:
        """
        Refines the plan based on feedback by re-prompting the ReactAgent.
        """

        if not feedback:
            return plan  # No feedback, no refinement needed
        
        for k, v in feedback.items():
            print("\nREFINING PLAN\n")
            refinement_prompt = f"Current plan:\n{plan}\n\nImprove the plan for the included query based on the attached feedback. Do not change aspects outside of the feedback.\n" + "query: \n" + self.query + "\nfeedback:\n " + k + ": " + v
            refined_plan, _, _ = self.react_agent.run(refinement_prompt, reset=True) # Do I want true or false? True new agent, false keeps mems+ context.
        
        return refined_plan

    def run(self, query: str) -> str:
        """
        Runs the manager agent to generate and refine a travel plan.
        """
        print("\nRunning ManagerAgent...")
        plan, _, _ = self.react_agent.run(query)
        print("\nInitial Plan:", plan)

        for i in range(self.max_iterations):
            evaluations = self.evaluate_plan(plan)

            plan = self.refine_plan(plan, evaluations)

        print("\nManagerAgent finished.")
        return plan

if __name__ == '__main__':
    # Example usage
    tools_list = ["notebook", "flights", "attractions", "accommodations",
                  "restaurants", "googleDistanceMatrix", "planner", "cities"]
    query = "Can you create a 3-day travel plan for 2 people from Orlando to Boston from March 15th to March 17th, 2022, with a budget of $2000?  The user prefers vegetarian options and wants to stay in a hotel."
    manager = ManagerAgent(query=query, tools=tools_list)
    final_plan = manager.run(query)
    print("Final Plan:", final_plan)