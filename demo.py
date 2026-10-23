"""
Demo script for the multi-agent research system.
Runs with mock data for testing purposes.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def setup_mock_environment():
    """Setup mock environment variables for demo."""
    os.environ["GOOGLE_API_KEY"] = "demo_key_for_testing"
    os.environ["OPENALEX_EMAIL"] = "demo@example.com"
    os.environ["RESEARCH_DOMAIN"] = "computer_science"
    os.environ["MAX_ITERATIONS"] = "5"
    os.environ["SUPERVISOR_THRESHOLD"] = "7.0"
    os.environ["DEBUG_MODE"] = "true"
    os.environ["LOG_LEVEL"] = "INFO"

def create_mock_agents():
    """Create mock agent implementations for demo."""
    
    # Mock TopicHunterAgent
    class MockTopicHunterAgent:
        def __init__(self):
            pass
            
        def discover_topics(self, domain):
            return [
                {
                    "title": "Novel Approaches to Transformer Architecture Optimization",
                    "description": "Exploring new methods to improve transformer efficiency",
                    "feasibility": 8,
                    "score": 9.2,
                    "rank": 1
                },
                {
                    "title": "Multi-Modal Learning for Scientific Literature Analysis",
                    "description": "Combining text and visual data for better research insights",
                    "feasibility": 7,
                    "score": 8.5,
                    "rank": 2
                }
            ]
    
    # Mock HypothesisDebateSystem
    class MockHypothesisDebateSystem:
        def __init__(self):
            pass
            
        def conduct_debate(self, topic):
            from agents.hypothesis_debate import DebateResult
            # Simulate first topic failing, second topic passing
            if "Transformer" in topic["title"]:
                return DebateResult(
                    topic=topic["title"],
                    proposer_argument="Strong theoretical foundation and clear methodology",
                    challenger_argument="Some concerns about scalability",
                    moderator_decision="FAIL",
                    score=4.5,
                    passed=False,
                    reasoning="Concerns about scalability and implementation complexity"
                )
            else:
                return DebateResult(
                    topic=topic["title"],
                    proposer_argument="Strong theoretical foundation and clear methodology",
                    challenger_argument="Some concerns about scalability",
                    moderator_decision="PASS",
                    score=8.5,
                    passed=True,
                    reasoning="Strong theoretical foundation and clear methodology"
                )
    
    # Mock PlannerAgent
    class MockPlannerAgent:
        def __init__(self):
            pass
            
        def create_plan(self, topic):
            return {
                "title": topic["title"],
                "sections": ["Abstract", "Introduction", "Related Work", "Methods", "Experiments", "Conclusion"],
                "experiments": [
                    {"name": "baseline_comparison", "type": "comparison", "description": "Compare with existing methods"},
                    {"name": "ablation_study", "type": "analysis", "description": "Analyze component contributions"}
                ],
                "timeline": "6 months",
                "resources": ["GPU cluster", "Academic datasets"]
            }
    
    # Mock WriterAgent
    class MockWriterAgent:
        def __init__(self):
            pass
            
        def draft_section(self, section_name, topic, plan, engineer_outputs):
            mock_content = {
                "Abstract": "This paper presents a novel approach to transformer architecture optimization...",
                "Introduction": "Transformer models have revolutionized natural language processing...",
                "Related Work": "Previous work in transformer optimization includes attention mechanisms...",
                "Methods": "Our approach combines several innovative techniques...",
                "Experiments": "We conducted extensive experiments on multiple datasets...",
                "Conclusion": "In this work, we presented a novel method for transformer optimization..."
            }
            return mock_content.get(section_name, f"Content for {section_name} section...")
    
    # Mock SupervisorAgent
    class MockSupervisorAgent:
        def __init__(self):
            pass
            
        def evaluate_section(self, section_name, content, **_kwargs):
            # Return mock scores
            mock_scores = {
                "Abstract": 8.5,
                "Introduction": 8.0,
                "Related Work": 7.5,
                "Methods": 8.2,
                "Experiments": 7.8,
                "Conclusion": 8.0
            }
            score = mock_scores.get(section_name, 7.0)
            feedback = f"Good content for {section_name}. Score: {score}/10"
            return score, feedback
    
    # Mock EngineerAgent
    class MockEngineerAgent:
        def __init__(self):
            pass
            
        def run_experiment(self, experiment):
            raw_path = Path("./output/raw_results/demo_experiment.json")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps({"metrics": {"accuracy": 0.85, "speed": 0.92}}), encoding="utf-8")
            return {
                "experiment_name": experiment.get("name", "unknown"),
                "success": True,
                "results": {"accuracy": 0.85, "speed": 0.92},
                "aggregate_metrics": {"accuracy": {"mean": 0.85, "std": 0.02}},
                "raw_results_path": str(raw_path),
                "code": "import json\nprint(json.dumps({'metrics': {'accuracy': 0.85}}))",
                "visualizations": ["performance_comparison.png"],
                "timestamp": "2024-01-01T00:00:00"
            }
    
    # Mock EditorAgent
    class MockEditorAgent:
        def __init__(self):
            pass
            
        def create_final_paper(self, topic, draft_sections, plan, engineer_outputs, **_kwargs):
            return {
                "title": topic["title"],
                "sections": draft_sections,
                "experiments": engineer_outputs,
                "plan": plan
            }
        
        def generate_latex(self, final_paper):
            latex_template = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{graphicx}

\title{""" + final_paper["title"] + r"""}

\begin{document}
\maketitle

\begin{abstract}
""" + final_paper["sections"].get("Abstract", "") + r"""
\end{abstract}

\section{Introduction}
""" + final_paper["sections"].get("Introduction", "") + r"""

\section{Related Work}
""" + final_paper["sections"].get("Related Work", "") + r"""

\section{Methods}
""" + final_paper["sections"].get("Methods", "") + r"""

\section{Experiments}
""" + final_paper["sections"].get("Experiments", "") + r"""

\section{Conclusion}
""" + final_paper["sections"].get("Conclusion", "") + r"""

\end{document}
"""
            return latex_template
    
    # Mock MetaAgent
    class MockMetaAgent:
        def __init__(self):
            pass
            
        def evaluate_system_performance(self, state):
            return "System performing well. Good progress on research topic."
        
        def should_reset(self, state):
            return False
        
        def should_continue(self, state):
            return True
    
    return {
        "TopicHunterAgent": MockTopicHunterAgent,
        "HypothesisDebateSystem": MockHypothesisDebateSystem,
        "PlannerAgent": MockPlannerAgent,
        "WriterAgent": MockWriterAgent,
        "SupervisorAgent": MockSupervisorAgent,
        "EngineerAgent": MockEngineerAgent,
        "EditorAgent": MockEditorAgent,
        "MetaAgent": MockMetaAgent
    }

def run_demo():
    """Run the demo with mock agents."""
    print("🤖 Multi-Agent AI Research System - DEMO MODE")
    print("=" * 60)
    print("This demo runs with mock data and agents for testing purposes.")
    print("No real API calls will be made.")
    print("=" * 60)
    
    # Setup mock environment
    setup_mock_environment()
    
    # Create mock agents
    mock_agents = create_mock_agents()
    
    # Monkey patch the agent modules before importing main
    import agents.topic_hunter
    import agents.hypothesis_debate
    import agents.planner
    import agents.writer
    import agents.supervisor
    import agents.engineer
    import agents.editor
    import agents.meta_agent
    
    # Replace the agent classes
    agents.topic_hunter.TopicHunterAgent = mock_agents["TopicHunterAgent"]
    agents.hypothesis_debate.HypothesisDebateSystem = mock_agents["HypothesisDebateSystem"]
    agents.planner.PlannerAgent = mock_agents["PlannerAgent"]
    agents.writer.WriterAgent = mock_agents["WriterAgent"]
    agents.supervisor.SupervisorAgent = mock_agents["SupervisorAgent"]
    agents.engineer.EngineerAgent = mock_agents["EngineerAgent"]
    agents.editor.EditorAgent = mock_agents["EditorAgent"]
    agents.meta_agent.MetaAgent = mock_agents["MetaAgent"]
    
    # Now import and run main
    import main
    
    # Run the main function
    try:
        # Override the workflow compilation to use higher recursion limit
        original_create_research_graph = main.create_research_graph
        def patched_create_research_graph():
            workflow = original_create_research_graph()
            return workflow
        
        main.create_research_graph = patched_create_research_graph
        
        exit_code = main.main()
        print(f"\n🎉 Demo completed with exit code: {exit_code}")
        return exit_code
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_demo()
    sys.exit(exit_code) 
