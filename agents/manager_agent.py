from typing import List
from .tool_agents import ReactAgent
from agents.local_model import LocalModel 

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
        self.llm = LocalModel(model_path="/scratch/gpfs/ca2992/models/DeepSeek-R1-Distill-Llama-8B")
        self.react_agent = ReactAgent(
            args=None,
            tools=tools,
            react_llm_name=react_llm_name,
            planner_llm_name=planner_llm_name,
            max_steps=30
        )
        print("\nPlanner LLM: ", planner_llm_name)
        print("\nReact LLM: ", react_llm_name)
        print("\n Manager LLM: ", self.llm.name)
        self.max_iterations = max_iterations
        
    def evaluate_plan(self, plan: str) -> str:  # Changed return type to Dict[str, str]
        """
        Evaluates the plan against the criteria and returns feedback.
        """
        
        prompt = "You are a manager agent evaluating a travel plan. Evaluate the plan on Budget Compliance, Room Rules, Room Type, Cuisine, Transportation, and Common Sense. Provide feedback for each specification, with reference to the user query. For example, if the plan is over budget, you need to include this in your feedback, as for the other categories."
        
        prompt = prompt + " The query is as follows: \n" + self.query + "\n\n"

        prompt = prompt + "The plan is as follows: \n" + plan + "\n\n"

        prompt = prompt + "If you find it meets all constraints, say PLAN APPROVED. Otherwise, provide feedback for each specification. \n"

        prompt = prompt + "If it is vague whether a criterion is met, prompt your agent to double-check on its next run while planning. \n"

        prompt = prompt + "Give Feedback: \n"

        evaluation = self.llm(prompt)
        
        return evaluation

    def refine_plan(self, plan: str, feedback: str) -> tuple[str, str]:
        """
        Refines the plan based on feedback by re-prompting the ReactAgent.
        """
        if len(feedback) == 0:
            print("potential error in agent, no feedback by manager.")
            return plan
        
        refinement_prompt = f"Current plan:\n{plan}\n\nImprove the plan for the included query based on the attached feedback. Do not change aspects outside of the feedback.\n" + "query: \n" + self.query + "\nfeedback:\n "
        refinement_prompt = refinement_prompt + feedback
        print("\nREFINING PLAN\n")
        refined_plan, _, json = self.react_agent.run(refinement_prompt, reset=True) # Do I want true or false? True new agent, false keeps mems+ context.
        
        return refined_plan, str(json)

    def run(self, query: str) -> tuple[str, str]:
        """
        Runs the manager agent to generate and refine a travel plan.
        """
        print("\nRunning ManagerAgent...")
        print("\nQuery:", query + "\n")
        plan, _, json = self.react_agent.run(query)
        print("\nInitial Plan:", plan, json)
        if (len(plan) == 0 and len(str(json)) == 0):
            print("ERROR: No initial plan generated. Exiting.")
            return plan, str(json)

        if (len(plan) == 0):
            plan = str(json)
        print("\nEvaluating Plan with ", self.max_iterations, " iterations.")
        for i in range(self.max_iterations):
            evaluations = self.evaluate_plan(plan)

            plan, json = self.refine_plan(plan, evaluations)

        print("\nManagerAgent finished.")
        return plan, str(json)

if __name__ == '__main__':
    # Example usage
    tools_list = ["notebook", "flights", "attractions", "accommodations",
                  "restaurants", "googleDistanceMatrix", "planner", "cities"]
    query = "Can you create a 3-day travel plan for 2 people from Orlando to Boston from March 15th to March 17th, 2022, with a budget of $2000?  The user prefers vegetarian options and wants to stay in a hotel."
    manager = ManagerAgent(query=query, tools=tools_list)
    final_plan, json = manager.run(query)
    print("Final Plan:", final_plan)
    print("JSON:", json)