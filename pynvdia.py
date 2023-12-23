from pynvml import *
def nvidia_info():
    # pip install nvidia-ml-py
    nvidia_dict = {
        "state": True,
        "nvidia_version": "",
        "nvidia_count": 0,
        "gpus": []
    }
    try:
        nvmlInit()
        nvidia_dict["nvidia_version"] = nvmlSystemGetDriverVersion()
        nvidia_dict["nvidia_count"] = nvmlDeviceGetCount()
        for i in range(nvidia_dict["nvidia_count"]):
            handle = nvmlDeviceGetHandleByIndex(i)
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            gpu = {
                "gpu_name": nvmlDeviceGetName(handle),
                "total": memory_info.total,
                "free": memory_info.free,
                "used": memory_info.used,
                "temperature": f"{nvmlDeviceGetTemperature(handle, 0)}℃",
                "powerStatus": nvmlDeviceGetPowerState(handle)
            }
            nvidia_dict['gpus'].append(gpu)
    except NVMLError as _:
        nvidia_dict["state"] = False
    except Exception as _:
        nvidia_dict["state"] = False
    finally:
        try:
            nvmlShutdown()
        except:
            pass
    return nvidia_dict

def check_gpu_mem_usedRate(list = None):
    max_rate = 0.0
    # while True:
    info = nvidia_info()
    # print(info)
    if_print = False
    if list is None:
        for i in range(info['nvidia_count']):
            used = info['gpus'][i]['used']
            tot = info['gpus'][i]['total']
            if if_print:
                print(f"GPU{i} used: {round(used/1024/1024,2)}MB, 使用率：{round(used/tot,2)}")
    elif type(list) == int:
        i = list
        used = info['gpus'][i]['used']
        tot = info['gpus'][i]['total']
        if if_print:
            print(f"GPU{i} used: {round(used/1024/1024,2)}MB, 使用率：{round(used/tot,2)}")       
    else:
        for i in list:
            used = info['gpus'][i]['used']
            tot = info['gpus'][i]['total']
            if if_print:
                print(f"GPU{i} used: {round(used/1024/1024,2)}MB, 使用率：{round(used/tot,2)}")


def memory(i):
    info = nvidia_info()
    used = info['gpus'][i]['used']
    return round(used/1024/1024,2)