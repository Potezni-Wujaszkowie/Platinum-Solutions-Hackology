from pytrends.request import TrendReq
from matplotlib import pyplot as plt

def get_product_popularity(product, num_years):
    pytrends = TrendReq(hl='en-US', tz=360)
    timeframe = f'today {num_years}-y'
    pytrends.build_payload([product], timeframe=timeframe)
    interest_over_time = pytrends.interest_over_time()
    interest_over_time_monthly = interest_over_time.resample('M').mean()
    popularity_list = interest_over_time_monthly[product].tolist()

    plt.figure(figsize=(10,6))
    plt.plot(interest_over_time_monthly.index, interest_over_time_monthly[product], label=product, color='b')
    plt.title(f'Google Search Interest Over Time for {product} (Last {num_years} Years)')
    plt.xlabel('Date')
    plt.ylabel('Popularity')
    plt.grid(True)
    plt.legend()

    plot_file = "popularity_plot.png"
    plt.savefig(plot_file)
    plt.show()


get_product_popularity("iPhone", 5)
