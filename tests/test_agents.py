from agents.agent import Agent
from agents.interval_agent import AgentInterval, EuclideanIntervalDistribution, UniformDistribution, GaussianDistribution
from agents.llm_agent import LLMAgent
import matplotlib.pyplot as plt
import seaborn as sns
from datasets.demographic_utils import sample_profiles, prepare_demographic_data


def test_agent_initialization(agent_id):
    agent = Agent(agent_id=agent_id)
    print(f"Name: {agent.name}, ID: {agent.agent_id}")


def test_agent_interval_with_uniform_distribution_initialization():
    # Agent creation

    ## No distribution is given but min , max is does
    a1 = AgentInterval(agent_id=1, min_score=0.5, max_score=0.6)
    print(f"Agent {a1.agent_id} distribution is {a1.distribution} with interval [{a1.min_score}, {a1.max_score}]")

    ## Uniformal

    # - only with agent index
    a2 = AgentInterval(agent_id=2)
    print(f"Agent {a2.agent_id} distribution is {a2.distribution} with interval [{a2.min_score}, {a2.max_score}]")
    # - distribution specified
    a3 = AgentInterval(agent_id=3, distribution=UniformDistribution())
    print(f"Agent {a3.agent_id} distribution is {a3.distribution} with interval [{a3.min_score}, {a3.max_score}]")


def test_agent_interval_with_gaussian_distribution_initialization():
    # - with default parameters: id and distribution
    a4 = AgentInterval(agent_id=4, distribution=GaussianDistribution())
    print(f"Agent {a4.agent_id} distribution is {a4.distribution} with interval [{a4.min_score}, {a4.max_score}]")

    # - with required parameters: mu, sigma
    a5 = AgentInterval(agent_id=5, distribution=GaussianDistribution(mu=0.25, sigma=0.05))
    print(f"Agent {a5.agent_id} distribution is {a5.distribution} with interval [{a5.min_score}, {a5.max_score}]")



def create_community(num_agents: int, distribution: EuclideanIntervalDistribution = None,
                     proportion_25: float = 0) -> list:
    """
    Create a community of agents with a specified distribution.

    :param num_agents: Number of agents to create.
    :param distribution: The distribution type to use for initializing agent intervals (default is uniform if None).
    :param proportion_25: Proportion of agents with a Gaussian distribution centered at 0.25 (optional, 0 means no split).
    :return: A list of AgentInterval objects.
    """
    agents = []

    # Two peaks Gaussian
    if proportion_25 > 0 and isinstance(distribution, GaussianDistribution):
        # Split the agents based on proportion_25 for two-peak Gaussian distribution
        num_agents_25 = int(num_agents * proportion_25)  # Number of agents around mu=0.25

        # Create agents with distribution centered at 0.25
        for i in range(1, num_agents_25 + 1):
            agents.append(
                AgentInterval(agent_id=i, distribution=GaussianDistribution(mu=0.25, sigma=distribution.sigma)))

        # Create agents with distribution centered at 0.75
        for i in range(num_agents_25 + 1, num_agents + 1):
            agents.append(
                AgentInterval(agent_id=i, distribution=GaussianDistribution(mu=0.75, sigma=distribution.sigma)))

    # Same distribution for all agents
    else:
        for i in range(1, num_agents + 1):
            agents.append(AgentInterval(agent_id=i, distribution=distribution))

    return agents


def test_agent_interval_distributions():
    # - Uniformal
    communityA = create_community(num_agents=100000, distribution=UniformDistribution())
    for i in communityA:
        print(f"Agent {i.agent_id} distribution is {i.distribution} with interval [{i.min_score}, {i.max_score}]")

    # - Gaussian
    communityB = create_community(num_agents=100000, distribution=GaussianDistribution())
    for i in communityB:
        print(f"Agent {i.agent_id} distribution is {i.distribution} with interval [{i.min_score}, {i.max_score}]")

    # - Two peaks gaussian
    communityC = create_community(num_agents=100000, distribution=GaussianDistribution(), proportion_25=0.5)
    for i in communityC:
        print(f"Agent {i.agent_id} distribution is {i.distribution} with interval [{i.min_score}, {i.max_score}]")

    communities = [communityA, communityB, communityC]

    # Plot the middle point
    for c in communities:
        middle_points = [(agent.min_score + agent.max_score) / 2 for agent in c]
        plt.figure()
        plt.hist(middle_points, bins=10, color="#7f7f7f", edgecolor='black', alpha=0.7)
        sns.kdeplot(middle_points, color="blue", linewidth=2)
        plt.tight_layout()

    plt.show()


def test_llm_agent():
    # Agent creation
    agent_id = 1
    profile = {
        "Sex": "Male",
        "Age": "35",
        "Educational attainment": "Upper secondary education",
    }
    topic = "climate change policy"
    topic_position = 0.45
    agent = LLMAgent(agent_id=agent_id, profile=profile, topic=topic, topic_position=topic_position)
    # Community creation
    df_prepared = prepare_demographic_data()
    sampled_profiles = sample_profiles(df_prepared, num_samples=10)
    agents = LLMAgent.create_community(sampled_profiles, topic="climate change policy")
    print("The agents community created successfully")
    for agent in agents:
        print(f"Agent ID: {agent.agent_id}, Topic Position: {agent.topic_position}")
        print(f"Profile: {agent.profile}")
        print("-" * 50)
    # Generate system prompt
    system_prompt = agent.construct_system_prompt()
    print("System Prompt:\n")
    print(system_prompt)
    #
    mock_current_state = """
    Paragraph ID | Text                                   | + Votes | - Votes | Own Vote
    -------------------------------------------------------------------------------------
    1            | Implement a carbon tax on emissions    | 2       | 1       | -1
    2            | Subsidize electric vehicle purchases   | 3       | 2       | ?
    3            | Increase investment in solar energy    | 4       | 0       | +1
    """.strip()
    prompt = agent.construct_decision_prompt(mock_current_state)
    print(prompt + "\n")
    # Action
    action = agent.decision_chat(mock_current_state)
    print("Agent action:\n")
    print(action)
    # Sampling
    print(agent.grouped_examples)
    print(agent.sample_diverse_proposals())


def test_api_key():
    """Test that the API key is loaded correctly from .env and works with OpenAI."""
    print("Testing API key from .env file...")
    from dotenv import load_dotenv
    import os
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    # Load .env file
    load_dotenv()

    # Check if API key exists
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment variables.")
        return False

    # Show masked key for verification
    masked_key = f"sk-...{api_key[-4:]}" if api_key.startswith("sk-") else "Invalid key format"
    print(f"Found API key: {masked_key}")

    try:
        # Test API connection
        print("Testing connection to OpenAI API...")
        chat = ChatOpenAI()
        response = chat.invoke([HumanMessage(
            content="Hello! Just testing the API connection. Please reply with a simple 'API test successful'.")])

        print(f"Response from API: {response.content}")
        answer = True if "successful" in response.content.lower() else False
    except Exception as e:
        print(f"ERROR connecting to OpenAI API: {str(e)}")
        answer = False
    print(f"Successful? {answer}")




if __name__ == "__main__":
    # print("Running tests for agents...")
    # test_agent_initialization(agent_id=5)
    # test_agent_interval_with_uniform_distribution_initialization()
    # test_agent_interval_with_gaussian_distribution_initialization()
    # test_agent_interval_distributions()
    # test_api_key()
    test_llm_agent()


