from typing import List, Dict, Any
from agents.tool_agents import ReactAgent  # Import your ReactAgent
from prompts import MANAGER_AGENT_INSTRUCTION

class ManagerAgent:
    def __init__(self,
                 tools: List[str],
                 react_llm_name: str = 'gpt-3.5-turbo-1106',
                 planner_llm_name: str = 'gpt-3.5-turbo-1106',
                 max_iterations: int = 3,
                 evaluation_criteria: List[str] = None):  # Add criteria
        """
        Initializes the ManagerAgent.

        Args:
            tools (List[str]): List of tools for the ReactAgent.
            react_llm_name (str): Name of the LLM for the ReactAgent.
            planner_llm_name (str): Name of the LLM for the planner tool.
            max_iterations (int): Maximum iterations for plan refinement.
            evaluation_criteria (List[str]): Criteria to evaluate the plan.
        """
        self.react_agent = ReactAgent(
            args=None,  
            tools=tools,
            react_llm_name=react_llm_name,
            planner_llm_name=planner_llm_name
        )
        self.max_iterations = max_iterations
        self.evaluation_criteria = evaluation_criteria or [
            "Feasibility",
            "Completeness",
            "Coherence"
        ]  # Default criteria

    def refine_plan(self, plan: str) -> str:
        """""
        Refines the plan based on feedback.

        Args:
            plan (str): The original travel plan.

        Returns:
            str: Feedback on the given plan
            
        """
        
        # Placeholder for refinement logic
        return plan  # Return the original plan for now

    def run(self, query: str) -> str:
        """
        Runs the manager agent to generate and refine a travel plan.

        Args:
            query (str): The user's travel query.

        Returns:
            str: The final travel plan.
        """
        plan, scratchpad, action_log = self.react_agent.run(query)
        print("Initial Plan:", plan)

        for i in range(self.max_iterations):
            evaluations = self.evaluate_plan(plan)
            print(f"Evaluation (Iteration {i+1}):", evaluations)

            if all(evaluations.values()):
                print("Plan meets all criteria.")
                return plan  # Plan is good

            # Generate feedback (replace with actual feedback generation)
            feedback = "The plan needs improvement."

            plan = self.refine_plan(plan, feedback)
            print(f"Refined Plan (Iteration {i+1}):", plan)

        print("Maximum iterations reached.")
        return plan  # Return the best plan found

if __name__ == '__main__':
    # Example usage
    tools_list = ["notebook", "flights", "attractions", "accommodations",
                  "restaurants", "googleDistanceMatrix", "planner", "cities"]
    manager = ManagerAgent(tools=tools_list)
    query = "I want to visit New York City for 3 days.  Suggest some attractions and restaurants."
    final_plan = manager.run(query)
    print("Final Plan:", final_plan)