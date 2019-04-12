---
title: The polls had systematic error but the model explained them well
author: Yitzhak Sapir
math: true
---
<style>
table { display: block; overflow-x: auto; white-space: nowrap }
</style>

The election is now over and the votes have been counted. We now even have official results and the polls were very clearly off. Indeed, the day after election day in Israel is usually called the pollsters' "Judgement Day." As a filler in the minutes leading to the broadcast of the exit polls, the Israeli satire show Eretz Nehederet even aired a skit showing veteran pollster Mina Tzemach chasing the voters who lie to her in the exit polls and using a duck and piñata in lieu of complex statistical models (in Hebrew):

<iframe width="632" height="473" src="https://www.mako.co.il/AjaxPage?jspName=embedHTML5video.jsp&galleryChannelId=770e3d99ade16110VgnVCM1000004801000aRCRD&videoChannelId=17ef630d7b20a610VgnVCM2000002a0c10acRCRD&vcmid=e03754d84530a610VgnVCM2000002a0c10acRCRD" frameborder="0" allowfullscreen></iframe>

I will discuss the reasons the polls are so often off in Israel in a different post. But here I want to discuss the performance of the model as a guide to understanding the polls.

<!--more-->

We might ask - what do we expect of a model? Ideally, we would somehow want the model to predict the most accurate results. But the model's basic data is always the polls themselves. The model could be adjusted for likely voters, fundamentals, and momentum and trends among undecided voters. But if there is systematic error in the polls, the prediction will likewise be off and any adjustment will only have a minor effect.

In light of this, what we might ask of a model is that it also present us the variability that we can expect. We know what the polls say, but how far from the polls can the actual result be?

## Pre-election Poll Errors
A common method of computing poll errors is to use a distance formula. Using a distance formula, the final pre-election polls erred by an average of 14.53 mandates:

Pollster|Publication|Error|B&W-Likud|Likud|Blue & White|Shas|UTJ|Labor|Hadash-Taal|Yisrael Beitenu|Right Union|Meretz|Kulanu|Raam-Balad|New Right|Zehut|Gesher
:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:
Final [(PDF)](https://bechirot21.bechirot.gov.il/election/Documents/%D7%93%D7%95%D7%91%D7%A8%20%D7%95%D7%A2%D7%93%D7%AA%20%D7%94%D7%91%D7%97%D7%99%D7%A8%D7%95%D7%AA/letter_results.pdf)|Election|0|-1|36|35|8|7|6|6|5|5|4|4|4|0|0|0
Midgam Proj. [(PDF)](https://bechirot21.bechirot.gov.il/election/Decisions/Documents/%D7%A1%D7%A7%D7%A8%D7%99%D7%9D/13news_5.4.19.pdf)|Ch. 13|15.02|-0.8|28.4|27.6|5|6|11|6|4|7|5|4|4|6|6|0
Smith [(Article)](https://www.maariv.co.il/elections2019/polls/Article-692912)|Maariv|14.00|2|27|29|6|7|9|7|4|6|5|5|4|6|5|0
Maagar Mochot [(PDF)](https://bechirot21.bechirot.gov.il/election/Decisions/Documents/%D7%A1%D7%A7%D7%A8%D7%99%D7%9D/radio103_5.4.19.pdf)|103FM|14.97|3|28|31|6|7|9|7|0|7|7|6|0|6|6|0
Panels [(Article)](https://elections.walla.co.il/item/3228485)|Walla|13.78|1|29|30|5|6|10|8|4|7|6|0|4|6|5|0
Midgam [(PDF)](https://bechirot21.bechirot.gov.il/election/Decisions/Documents/%D7%A1%D7%A7%D7%A8%D7%99%D7%9D/news_4.4.19.pdf)|Ch. 12|14.63|4|26|30|5|7|10|7|5|5|5|5|4|6|5|0
TNS [(Article)](https://twitter.com/kann_news/status/1113853811330822145)|Kann|13.34|-1|31|30|4|6|8|8|0|6|6|5|4|6|6|0
Smith [(Article)](https://www.jpost.com/Israel-News/Post-poll-predicts-easy-win-for-Right-585821)|Jerusalem Post|14.28|1|27|28|6|6|9|6|5|6|5|5|4|5|4|4
Midgam [(PDF)](https://bechirot21.bechirot.gov.il/election/Decisions/Documents/%D7%A1%D7%A7%D7%A8%D7%99%D7%9D/yedioth_4.4.19.pdf)|Yediot Achronot|14.97|4|26|30|5|7|11|7|4|5|5|5|4|6|5|0
Maagar Mochot [(PDF)](https://bechirot21.bechirot.gov.il/election/Decisions/Documents/%D7%A1%D7%A7%D7%A8%D7%99%D7%9D/israel_hayom_4.4.19.pdf)|Yisrael Hayom|15.81|5|27|32|5|8|10|6|0|6|8|6|0|6|6|0

The error of the various final pre-election polls ranges from 13.34 to 15.02 and averages at 14.53.

## How the Model Performed
With the final results, we can now also compare the model's forecast to the polls and to another forecast by Knesset Jeremy. For comparison, I'm also adding here the simple average of all pre-election polls:

Forecast|Error|B&W-Likud|Likud|Blue & White|Shas|UTJ|Labor|Hadash-Taal|Yisrael Beitenu|Right Union|Meretz|Kulanu|Raam-Balad|New Right|Zehut|Gesher
:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:
Final|0|-1|36|35|8|7|6|6|5|5|4|4|4|0|0|0
Poll Average|13.85|2.02|27.71|29.73|5.22|6.67|9.67|6.89|2.89|6.11|5.78|4.56|3.11|5.89|5.33|0.44
[pyHoshen](https://pyhoshen.org/2019/04/07/2019-Election-Final-Forecast.html)|13.86|3|28|31|6|7|10|8|0|6|4|5|4|6|5|0
[Knesset Jeremy](https://knessetjeremy.com/2019/04/07/knessetjeremy-phase-2-prediction-analysis/)|12.49|-1|30|29|5|6|9|7|4|6|5|4|4|6|5|0

So first, hats off to Knesset Jeremy whose prediction beat all the polls. Its "distance" is 12.49 about one mandate less than the closest poll by TNS at 13.34. It also correctly predicted that Likud would have a 1 mandate advantage over Blue & White!

But my model's prediction also was relatively good. Its average error closely represents the error of the poll average at approximately the average error of all the final pre-election polls. We can see this in the graph below:

![Final Pre-election Poll Errors and the Model](/images/2019-04-12-The-model-explained-the-polls-well/2019-04-12-The-model-explained-the-polls-well-poll-and-model-errors.png)

The errors for the various polls are plotted above in blue with the average error at 14.53 in dark blue. The error of pyHoshen is slightly better than this, and Knesset Jeremy's even more.

We can also explain pyHoshen's conclusions: The model predicted Yisrael Beitenu would not pass, and this was based on two pollsters (Maagar Mochot and TNS) which showed it below the threshold. It had Raam Balad passing because only one pollster (Maagar Mochot) had them below the threshold. It also had Blue and White passing Likud by 3 mandates. Indeed the average difference is about 2 across the polls.

## The Confidence Interval
But, as we mentioned above, even if the polls have systematic error which will affect our prediction, it would be nice to have an indication of how far the true result could differ from the prediction. The model provided an indication in the form  of 95% confidence intervals for the results:

&#xfeff;|Likud|Blue & White|Shas|UTJ|Labor|Hadash-Taal|Yisrael Beitenu|Right Union|Meretz|Kulanu|Raam-Balad|New Right|Zehut|Gesher
:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:
Final|36|35|8|7|6|6|5|5|4|4|4|0|0|0
pyHoshen 95%|23-34|25-36|0-10|4-11|5-14|0-12|0-7|0-12|0-9|0-9|0-8|0-11|0-10|0-5

For Gesher and Yisrael Beitenu which were predicted not to pass, the interval was not written on the graph but this is the value it would have computed. Except for the Likud which is 2 mandates beyond the 95% confidence interval, the model has all the final results within its confidence intervals. This is pretty amazing - the model predicts an outcome that is approximately the average of where all the polls lie from the final result. But it also correctly covers (almost) all the final results in its confidence intervals.

At the time it might have seemed the confidence intervals were a bit too extreme -- it predicted only four parties to be definitely clear of the threshold! But it's now clear that having 5 as a low end for Labor, or 34-36 as the high end for Likud and Blue and White was actually necessary. So as a tool to understand the polls, it did very well -- giving us both the average, and covering the expected variability.

The model also had Netanyahu's potential right-wing coalition partners at an average of 66 mandates (as did Knesset Jeremy). The final right wing partners for the coalition number at 65 mandates.

## Conclusions
The final prediction of the pyHoshen model was based on the polls of 48 days since Feb 21, 2019 when the party lists were finalized. Based on these polls it also computed a correlation matrix and "pollster house effects." While we do not have any way (independent of the model) to measure these, the correlation matrix did correctly find some conclusions such as the negative correlation between Hadash-Taal and Raam-Balad.

It is disappointing that we have such a severe systematic error and as a result such a high variability. The high variability indicating that the Likud was expected anywhere between 25 and 34 or the New Right anywhere between 0 and 11 mandates leave us with much uncertainty. But this is not due to the model which simply told us the correct uncertainty to expect so much as the polls themselves. The polls' poor performance (yet again) is a separate issue that has to do with the quality of the polls. Furthermore, the fact that the Likud's 36 mandates is not covered by the 95% confidence interval and that several other parties poll at the edges of the 95% confidence interval is probably an indication that the systematic polling error is more than simply "voters changing their minds" and probably indicates an issue with the underlying pollster methodology.

Despite the poor poll performance, we had much better information this year than in previous years as to what the polls actually said due to legal challenges against the polling firms. For example, in previous, parties polled below the threshold would be presented as "0" if at all. This year, the pollsters gave us the support percentage they polled those parties. For example, Yisrael Beitenu got 4.02% and 5 mandates in the final results. Final pre-election polls had it borderline. Some gave it 5 but others gave it 4 or 0. But we also knew that those who polled it at 0 mandates actually had it about halfway to the threshold at 3.25%. Maagar Mochot polled them at 2% while TNS polled them at 2.4%. This is information we simply do not have for previous election cycles and it allowed the model to predict Yisrael Beitenu at a 37% chance of passing the threshold.

So while it is clear and disappointing that the polls were all systematically off, the model was able to provide us with a good understanding of what they mean and what we can expect. The bottom line statements such as the Likud being in position to get between 25 to 34 mandates or Blue & White in position to get between 25 and 36 mandates, and the high chance that Netanyahu will be able to form a narrow right-wing government were correct. It gave us an "average" outcome that correctly summed up the polls, gave us a measure of variability that told us how far we can expect the real results to be, and provided us with a good idea of what kind of coalition we can expect to see.
