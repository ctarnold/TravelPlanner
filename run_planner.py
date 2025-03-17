from agents.manager_agent import ManagerAgent

def main():
    # Initialize tools and manager
    tools_list = ["notebook", "flights", "attractions", "accommodations",
                  "restaurants", "googleDistanceMatrix", "planner", "cities"]

    
    # Test query
    query = "Can you create a 3-day travel plan for 2 people leaving from Orlando and vacationing in Boston from March 15th to March 17th, 2022, with a budget of $2000?"
    
    manager = ManagerAgent(query = query, tools=tools_list)
    final_plan = manager.run(query)
    print("Final Plan:", final_plan)

if __name__ == "__main__":
    main()