#TO-DO: include pictures(espcially for Results) and tables in the final report.

Annotation Analysis of the Preliminary Dataset

Purpose:
As part of this project, we had to analyse manual annotations about hair density in images made by our preliminary groups. Our goal was to assess annotation quality, measure inter-rater agreement, and discover patterns, inconsistencies or bias in the annotations. This was a foundational task for our group before moving on to the final dataset.

Dataset:
During the mandatory assignment phase, each group manually annotated hair density on a subset of a dataset of images. As all 5 members of our team were in different groups, we could use all annotations from the 5 different groups. Overall, we had approximately 600 photos annotated over 5 groups. The annotators each assigned a discrete rating for an image on a scale of:
- 0 = No hair
- 1 = Some hair
- 2 = A lot of hair
In each group there were 4 to 5 annotators.

Methods:
To assess the qualities of the annotations we used different methods:
- Fleiss' Kappa Score: To assess inter-rater agreement in each group.
- Majority label calculation: Calculated which label has majority and plotted its distribution.
- Conflict analysis: Analysed which label conflicts occur the most in non-majority disagreements.
- Per-Rater Distribution: Visualized how often each annotator picked each label.
- Pairwise Rater Agreement(Confusion Matrix): Compared annotatators within a group to each other pairwisely to assess agreements and disagreements.

Results:
We experienced a fair amount of agreement, most annotators agreed on hair amount. However, there are groups which are less accurate and there are many images which caused confusion amongst annotators.

1.  Fleiss' Kappa scores: 
    - Ranged from moderate to substantial:
    - #O     0.708008
    - G     0.699370
    - #J     0.586344
    - #B     0.880848
    - #N     0.709016
    - #Interpretation:
    - #κ < 0.20: Slight
    - #0.21–0.40: Fair
    - #0.41–0.60: Moderate
    - #0.60: Substantial
    This suggests most annotator groups had good alignments. We picked out Group G and Group B to analyse further as they have the lowest and highest Fleiss' Kappa scores.
2. Disagreement Patterns:
    - Most images had a clear majority rating. 
    - In no-majority cases the most frequent conflict types were between 1 and 2 followed by 0 and 1.
3. Per-Rater Distributions:
    - We can notice different preferences in using certain ratings.
    - Some annotators favour specific ratings, which could indicate personal bias or different perspectives.
4. Pairwise Confusion Matrices:
    - High agreement between most raters (often >70%).
    - Most disagreements were mild (Δ = 1) rather than strong (Δ = 2).
    - From this we can see certain differences between groups either due to bias/different perspectives or less attention given by certain groups.

See Appendix for all heatmaps and rating distributions.

Discussion
We learned useful information about how subjective manual annotations can be:
- Annotating hair amount resulted in different opinions many times. As the rating scale is not concrete and upto interpretation annotators interpret it differently. 
- Disagreements were most common between adjacent categories which also strenhens our last claim.
- Some groups had stronger inter-rater consistency, potentially because of better communication, more awareness. Even though, we also have to take into consideration the number of raters and the number of images rated by each group.
- Alternatives: Expert annotatiors or automated
