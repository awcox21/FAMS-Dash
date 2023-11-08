from random import shuffle
from FAMS.model_rankings import Technology, Order, Ranking

num_metrics = 5
num_techs = 15
num_samples = num_techs

""" Generate and process rankings """
technologies = [Technology(id_=i) for i in range(num_techs)]
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
    ranking.to_json(f'{metric}.json')
