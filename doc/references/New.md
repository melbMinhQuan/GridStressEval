New government policy is set to increase the cost of distribution networks by encourag-
ing highly concentrated use during the middle of the day to absorb solar generation. This
document proposes a collaboration between university and distribution providers (DNSPs) to
explore an option to reduce the cost of this policy while retaining its benefits, to present to
the government while the policy is still in its trial phase.
The policy allows free electricity for a fixed window during peak solar. This causes prob-
lems on cloudy days if two heavy users share a phase on a distribution line. An expensive
solution is to rewire the customers to different phases, or upgrade the network to allow the
high current draw. The proposed cheaper solution is to allow a small number of customers to
be switched to non-overlapping windows, with no need for network modification.
The main contribution sought from DNSPs is anonymized data to evaluate the effectiveness
of the proposed solution, and advice about the specific nature of the problems foreseen by the
DNSPs that will need to be addressed.
1 Introduction
The federal government has recently mandated that electricity retailers offer a basic tariff with
free electricity for three hours in the middle of the day. The goals of this are three-fold:
1. To shift load to times when solar energy is abundant.
2. To share the benefits of solar generation to those who cannot install their own solar panels.
3. To be very simple for consumers to understand, and convenient for them to use.
However, this scheme risks imposing excessive loads on distribution networks during the period
of free electricity. Enhancing the network to accommodate these loads will be expensive. This
report proposes a minor modification that substantially reduces the cost and improves the ability
to use solar generation, with minimal impact on simplicity and convenience.
The proposal is to allow distribution network service providers (DNSPs) to shift the time of
free electricity for a small number of “heavy hitter” customers who draw excessive power during
the free period. These customers may, for example, be those who have two level-2 electric vehicle
chargers, able to draw 40 kW.
The premise is that there are many distribution circuits that can support one heavy hitter,
but cannot support two on cloudy days with minimal solar generation. The proposal is to have
three time slots: the original time of 11am to 2pm, and two new non-overlapping slots, say 9am–
noon and noon–3pm. Most customers would remain in the middle slot, which matches peak solar
generation, but when two heavy hitters share a phase on a circuit, they would be moved to the
two non-overlapping slots.
As shown in Figure 1, if the power drawn from these heavy users is small compared with that
drawn from the aggregate of the smaller users, this provides a better match to solar generation
than having all customers in the midday slot.
1
Figure 1: Left: power draw with a single free time slot. Right: power draw with heavy hitters in
non-overlapping time slots.
Questions we want to answer can be divided into those we can answer, and those we want to
be able to answer.
We can answer questions such as:
1. If x% of customers are heavy customers, and there are a known number of customers per
distribution line, what is the probability that the proposed scheme will avoid the need for a
network upgrade?
2. If we know the peak power consumption of each customer, what threshold should we use to
define “heavy”?
3. Given a solar generation profile and customer use pattern, how much solar do we expect
to be soaked using each of the two schemes, or competing proposals such as capping the
amount of free electricity use?