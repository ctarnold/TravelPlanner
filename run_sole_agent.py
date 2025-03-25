from agents.tool_agents import ReactAgent

def main():
    # Initialize tools and manager
    tools_list = ["notebook", "flights", "attractions", "accommodations",
                  "restaurants", "googleDistanceMatrix", "planner", "cities"]

    
    # Test query
    query = "Can you create a 3-day travel plan for 2 people leaving from Orlando and vacationing in Boston from March 15th to March 17th, 2022, with a budget of $2000?"
    
    agent = ReactAgent(args=None, tools=tools_list, 
                       react_llm_name='local',
                       planner_llm_name='local',
                       max_steps=30)

    iterations = 0
    while True:
        final_plan, _, json = agent.run(query)
        if final_plan is not None:
            break
        iterations += 1
        if iterations > 10:
            print("too many iterations, check agent", flush=True)
            break
    
    print("Final Plan:", final_plan)
    print("JSON:", json)

if __name__ == "__main__":
    main()