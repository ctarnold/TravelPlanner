from agents.manager_agent import ManagerAgent

def main():
    # Initialize tools and manager
    tools_list = ["notebook", "flights", "attractions", "accommodations",
                  "restaurants", "googleDistanceMatrix", "planner", "cities"]

    
    # Test query
    query = "Please create a 3-day travel itinerary for one person, beginning in Dallas and ending in Indianapolis between March 25th and March 27th, 2022. My budget for this trip is $1,300."
    
    manager = ManagerAgent(query = query, tools=tools_list)
    final_plan, _ = manager.run(query)
    print("Final Plan:", final_plan)
    # print("JSON:", json)

if __name__ == "__main__":
    main()