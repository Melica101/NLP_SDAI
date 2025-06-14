import jade.core.Profile;
import jade.core.ProfileImpl;
import jade.core.Runtime;
import jade.wrapper.AgentContainer;
import jade.wrapper.AgentController;
import jade.wrapper.StaleProxyException;

public class MainContainerLauncher {
    public static void main(String[] args) {
        // initialize JADE runtime
        Runtime rt = Runtime.instance();
        Profile p = new ProfileImpl();
        AgentContainer mainContainer = rt.createMainContainer(p);

        try {
            // create 5 NewsSpreader agents
            for (int i = 1; i <= 5; i++) {
                AgentController spreader = mainContainer.createNewAgent(
                    "NewsSpreaderAgent" + i,
                    "agents.NewsSpreaderAgent",
                    null
                );
                spreader.start();
            }

            // create 3 FactChecker agents
            for (int i = 1; i <= 3; i++) {
                AgentController checker = mainContainer.createNewAgent(
                    "FactCheckerAgent" + i,
                    "agents.FactCheckerAgent",
                    null
                );
                checker.start();
            }

            // create 2 Troll agents
            for (int i = 1; i <= 2; i++) {
                AgentController troll = mainContainer.createNewAgent(
                    "TrollAgent" + i,
                    "agents.TrollAgent",
                    null
                );
                troll.start();
            }

            System.out.println("All agents started successfully.");

        } catch (StaleProxyException e) {
            e.printStackTrace();
        }
    }
}
