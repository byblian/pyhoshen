---
title: A fundamentals component could improve the current model
author: Yitzhak Sapir
math: true
---
<style>
table { display: block; overflow-x: auto; white-space: nowrap }
</style>
Note: This post discusses takeaways from 2019. The Final Forecast for 2019 is [available and discussed here](http://pyhoshen.org/2019/04/07/2019-Election-Final-Forecast.html).

It has been almost two months since the elections. The polls missed quite a few details, indicating a systematic error. However, I was [happy with the model](https://pyhoshen.org/2019/04/12/The-polls-had-systematic-error-but-the-model-explained-them-well.html) - for a polls-only model it captured the both the poll average and gave a good idea of the final result in its margin of error.

It took me 3 months to fully develop the model. I started experimented with the system around Jan 15, 2019. It took about a month to get a converging model that captured the polls. I continued improving on the model over two months fixing various bugs. 

By this time, one month ahead of the elections, I considered adding a fundamentals component. Such a fundamentals model is [described here (PDF)](http://lukas-stoetzer.org/assets/forecast-multiparty_appendix.pdf). A fundamentals model has the potential to improve on the results of a polls-only model. Due to the lack of time, I also reviewed the results of fundamentals models on sites such as 538 - The results of the polls + fundamentals models were very close to the polls-only model, and did not seem to do much better.

Thus, due to both lack of time and the apparent little impact on the final results, I decided to focus on the visualization aspects. For example, in the last month, I added the coalitions graph and added colors to indicate failure to reach the threshold or to pass 60 members in the coalition, which can be crucial to actually winning the election.

## A Fundamentals Model

Having now passed the elections, it is time to consider how this model can perform better. In Israel, I believe a fundamentals model can provide an improvement. A [recent study](https://twitter.com/amit_segal/status/1126732393845366784) by Amit Segal and Nehemia Gershuni-Aylho shows that the voting patterns in the major voting blocs is extremely static. Comparing the vote in 2015 and 2019 in various major cities, the voting bloc representation was the same (&plusmn;1%). This is exactly what a fundamentals model would capture - the historical patterns that indicate how people vote based on their demographics.

A second reason to be optimistic about the contribution of the fundamentals model in Israel is the "poll blackout" period. In the USA, where polls run up to election day, the final forecast is largely based on the polls-only model with the fundamentals providing only a very small contribution. But the "poll blackout" period means that the fundamentals model would have a larger impact on the final forecast.

## The Challenge

The challenge in a fundamentals-based model in Israel is that the parties might change from election to election. I will try to find various ways to overcome this &mdash; for example, each party might have various attributes, demographic (% candidates who are secular/religious/ultra-religious, army veterans, arabs, men/women, etc.) and other (was a party in last elections, % candidates who were in the previous Knesset, who were previously Knesset members, who were previously politicans).

The results can then be used to see if the model correctly evaluates the April 2019 elections and the 2015 elections, for calibration.