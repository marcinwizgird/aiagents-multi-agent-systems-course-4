
**Teacher:** "Let's introduce a concept called **orchestration**. Imagine the system like a conductor leading an orchestra. The conductor (the orchestrator) coordinates different musicians (agents) to create a beautiful piece of music (a smooth operation). Each musician has a specific role and they work together to achieve a common goal. So, in our skate park system, we'll have several agents working for us. Each will handle specific tasks. For example, We'll have:"

*   **Customer Support Agent:** Handles customer inquiries and issues.
*   **Inventory Agent:** Manages shop stock, including skateboards, apparel, etc.
*   **Park Management Agent:** Deals with park bookings, events, and maintenance requests.

**(Teacher points to the code where the classes are defined)**

**Teacher:** "We'll be building the skeleton of how this all works together in this demo. In the exercise after, you will take this demo and build on it"

**(Teacher starts explaining the code, one section at a time)**

**Teacher:** "Let's start with the **CustomerSupportAgent**. Here is a basic class that represents a customer service representative. They can respond to different issues raised by a customer."

**(Teacher points to the `CustomerSupportAgent` class)**

**Teacher:** "First, let's look at the `__init__` method, it sets up our CustomerSupportAgent and give it a default name. Now, Let's look at the `diagnose_issue` method. Imagine a customer types a request. This method analyzes that request, looking for keywords that will help categorize the issue. For instance, If the customer says 'I want to buy a board' the system identifies it as an inquiry about a skateboard. If the customer asks about a session, it's linked to park booking. If they report a broken item, the system identifies it and understands the request."

**(Teacher explains the `provide_initial_response` method)**

**Teacher:** "The `provide_initial_response` method, then, gives the customer an initial response based on the diagnosis. Based on the customer's request, it gives an immediate response. In this example, the customer could learn from the agent's response or be routed to another agent."

**(Teacher moves on to the `InventoryAgent` class)**

**Teacher:** "Now, let's talk about our **InventoryAgent**. This agent is responsible for keeping track of what items are in stock at the shop. It has its own `__init__` method. Inside, we have a `stock` dictionary that holds the current stock levels for different items."

**(Teacher points to the `check_stock` method)**

**Teacher:** "And look at this! The `check_stock` method takes an item, checks our inventory dictionary, and tells us how many of that item are available. A very basic function."

**(Teacher moves on to the `ParkManagementAgent` class)**

**Teacher:** "Then there's the **ParkManagementAgent**. Here, we've a `get_booking_info` method. It provides basic information about booking sessions."

**(Teacher moves on to the `Orchestrator` class)**

**Teacher:** "Now, the most important part: the **Orchestrator**. This is our conductor. It brings everything together."

**(Teacher points to the `__init__` method of the `Orchestrator` class)**

**Teacher:** "The `__init__` method creates instances of our agents: the Customer Support Agent, the Inventory Agent and the Park Management Agent.  The Orchestrator is the main brain that handles requests."

**(Teacher points to the `handle_request` method)**

**Teacher:** "Here's the core logic in the `handle_request` method. First, it takes a user's request. Then, it passes that request to the `CustomerSupportAgent`'s diagnose method. This returns a diagnosis of the user's request. The customer support agent provides an initial response. Based on the diagnosis from the support agent, it directs the request to the appropriate agent: inventory or park management."

**(Teacher moves to the demo section)**

**Teacher:** "Now, let's see how it all works! We're going to run some tests. First is, let's make a booking"

**(Teacher runs the first request, showing the output)**

**Teacher:** "Here, the orchestrator passes the request to the Customer Support, which identifies it, and gives a response."

**(Teacher runs the second request, showing the output)**

**Teacher:** "Now, let's try asking about skateboards."

**(Teacher runs the third request, showing the output)**

**Teacher:** "Here, The Customer Support identifies it and provides information on how to follow up with the next agent."

**(Teacher pauses)**

**Teacher:** "This is a simplified model, but it shows the basic idea. Imagine this system scaled up: It could integrate with booking systems, track real-time inventory, even connect with payment gateways! This demo is designed to give you the core understanding for your next assignment."

**(Teacher concludes the demo)**

**Teacher:** "So, that's a taste of how orchestration can be applied in a real-world scenario like a Skate Park and Shop in Kenya! You can imagine it being a key part in a successful business"

**(Teacher looks towards the camera)**

**Teacher:** "In the coming exercise, you'll be expanding on this, adding features, improving the flow, and dealing with some more complex scenarios. Remember, this is just the beginning!  Keep experimenting, keep learning, and most importantly, keep that entrepreneurial spirit alive!"

**(Teacher smiles and waves goodbye)**
