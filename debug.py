import pandas as pd

# Example data
data = {
    'elapsed_time': [0, 20, 110, 200],
    'lat': [29.12, 29.13, 29.14, 29.16],
    'lon': [-94.78, -94.79, -94.77, -94.75]
}
segmented_df = pd.DataFrame(data)

# Set timedelta index
segmented_df = segmented_df.set_index(pd.to_timedelta(segmented_df['elapsed_time'], unit='s'))
segmented_df = segmented_df.drop(columns=['elapsed_time'])
segmented_df.index.name = 'elapsed_time'  # optional, just for clarity


# Resample
resampled = (
    segmented_df
    .resample('71s')
    .interpolate('linear')
    .reset_index(drop=False)
)



print(resampled)
