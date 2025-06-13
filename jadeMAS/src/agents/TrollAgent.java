package agents;

import jade.core.Agent;
import jade.core.behaviours.TickerBehaviour;
import jade.lang.acl.ACLMessage;
import jade.core.AID;

public class TrollAgent extends Agent {
    private final String[] fakeNewsSamples = {
            "Obama funds alien research in Area 51",
            "Senate passes immigration reform bill",
            "Biden signs executive order on AI safety",
            "Clinton behind secret child trafficking ring",
            "UN report on climate change sparks debate",
            "Pizzagate scandal resurfaces on social media",
            "White House denies moon landing conspiracy",
            "CDC warns of new flu outbreak this winter"
    };

    protected void setup() {
        addBehaviour(new TickerBehaviour(this, 5000) {
            protected void onTick() {
                String news = fakeNewsSamples[(int) (Math.random() * fakeNewsSamples.length)];
                ACLMessage msg = new ACLMessage(ACLMessage.INFORM);
                for (int i = 1; i <= 5; i++) {
                    msg.addReceiver(new AID("NewsSpreaderAgent" + i, AID.ISLOCALNAME));
                }
                msg.setContent(news);
                send(msg);
                // System.out.println(getLocalName() + " sent: " + news);
            }
        });
    }
}
