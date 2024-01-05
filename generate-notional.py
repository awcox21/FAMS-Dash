from random import shuffle, randint
from FAMS.model_rankings import Technology, Order, Ranking
from time import time

num_metrics = 5
num_techs = 15
num_categories = num_metrics // min((num_metrics // 2, 5))
num_samples = num_techs

""" Generate and process rankings """
t0 = time()
categories = [f'Category {i}' for i in range(num_categories)]
technologies = [Technology(id_=i,
                           category=categories[randint(0, num_categories - 1)])
                for i in range(num_techs)]
for metric_id in range(num_metrics):
    metric = f'Metric {metric_id}'
    samples = list()
    for _ in range(num_samples):
        sample = Ranking(technologies)
        techs = list(technologies)
        shuffle(techs)
        order = Order([[_] for _ in techs])
        sample.add_order(order)
        samples.append(sample)
    ranking = Ranking.combine(samples, name=metric)
    ranking.ranking_probabilities()
    ranking.to_json(f'metric-{metric}.json')
tf = time()

print(tf - t0)
