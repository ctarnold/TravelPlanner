from agents.manager_agent import ManagerAgent

def main():
    # Initialize tools and manager
    tools_list = ["notebook", "flights", "attractions", "accommodations",
                  "restaurants", "googleDistanceMatrix", "planner", "cities"]
    
    manager = ManagerAgent(tools=tools_list)
    
    # Test query
    query = "Can you create a 3-day travel plan for 2 people from Orlando to Boston from March 15th to March 17th, 2022, with a budget of $2000?"
    
    final_plan = manager.run(query)
    print("Final Plan:", final_plan)

if __name__ == "__main__":
    main()