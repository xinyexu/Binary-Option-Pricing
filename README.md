# Binary-Option-Pricing
European type Currency Binary Option Pricing with 3 methods and implied smile

### Data: 
•	(calls) PHLX (Philadelphia Stock Exchange (PHLX)) bid ask quotes for March XDE calls and puts on 1/22/08. <br />
•	(puts) PHLX bid ask quotes for March XDE calls and puts on 1/22/08. <br />
•	(euro) -  Daily $/€ 1/03/2000 – 1/22/2008 <br />
•	(libor) -  LIBOR rates EUR and USD. (22-Jan) <br />

### Option Description: 
The contract is a European call option, written on currency (dollar/euro FX), which has a payoff similar to a Heaviside step function, H(x).
Pricing date:  1/22/08 <br />
Underlying (1/22/208 contemporaneous with option quotes):  S0 = $/€ = 145.88 <br />
Derivatives:  European, Expiration date = 3/21/08, K = Strike price = 146 <br />

### Pricing Methods:
•	Analytical solution <br />
•	Binomial <br />
•	Monte Carlo <br />
•	Deriving implied volatility from the option prices

### Reference: 
1. Pricing formula: 
![alt text](https://github.com/xinyexu/Binary-Option-Pricing/blob/master/pricing%20formula.png)

Source: [Tomas Bjork] Arbitrage Theory in Continuous Time, Chpater 17.1

