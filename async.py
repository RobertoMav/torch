import asyncio
import random
import time


async def post_endpoint_call(api):
    time_results = []
    for i in range(4): # This is blocking the event loop from calling the api again before its response
        start_time = time.time()
        print(f"API {api} called at {start_time}")
        await asyncio.sleep(random.random() * 2)
        end_time = time.time()
        time_results.append(end_time - start_time)
        print(f"API {api} ran for {end_time - start_time}")
    return time_results


async def call_shit():
    results = await asyncio.gather(*(post_endpoint_call(api) for api in range(10)))
    return results



print("we are running some shit")
results = asyncio.run(call_shit())
t1 = "a"
t2 = "j"
t1 + t2
print(f"{t1 + t2=}")

print(f"{results=}")
print(f"{len(results)=}")
for result in results:
    print(f"{len(result)=}")