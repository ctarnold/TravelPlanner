from agents.manager_agent import ManagerAgent

def main():
    # Initialize tools and manager
    tools_list = ["notebook", "flights", "attractions", "accommodations",
                  "restaurants", "googleDistanceMatrix", "planner", "cities"]

    
    # Test query
    query = "Please plan a trip for me starting from Sarasota to Chicago for 3 days, from March 22nd to March 24th, 2022. The budget for this trip is set at $1,900"
    
    manager = ManagerAgent(query = query, tools=tools_list)
    final_plan, _ = manager.run(query)
    print("Final Plan:", final_plan)
    # print("JSON:", json)

if __name__ == "__main__":
    main()