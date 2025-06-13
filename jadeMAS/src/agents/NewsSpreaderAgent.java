package agents;

import jade.core.Agent;
import jade.core.behaviours.CyclicBehaviour;
import jade.lang.acl.ACLMessage;
import jade.lang.acl.MessageTemplate;
import jade.core.AID;

public class NewsSpreaderAgent extends Agent {
    protected void setup() {
        addBehaviour(new CyclicBehaviour() {
            public void action() {
                ACLMessage msg = receive(MessageTemplate.MatchPerformative(ACLMessage.INFORM));
                if (msg != null) {
                    String news = msg.getContent();
                    // System.out.println(getLocalName() + " received: " + news);

                    ACLMessage forward = new ACLMessage(ACLMessage.INFORM);
                    forward.setContent(news);
                    for (int i = 1; i <= 3; i++) {
                        forward.addReceiver(new AID("FactCheckerAgent" + i, AID.ISLOCALNAME));
                    }
                    send(forward);
                    // System.out.println(getLocalName() + " forwarded to FactCheckerAgents: " + news);
                } else {
                    block();
                }
            }
        });
    }
}