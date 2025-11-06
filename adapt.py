import subprocess
import csv
import os
import time
from datetime import datetime

# 数据集配置
DATASETS = ['SA', 'PU','TT']
#DATASETS = ['TT']
N_SP_RANGES = {
    #'IP': range(1100, 3601, 500),    # 1100,1600,2100,2600,3100,3600
    'SA': range(2700, 5201, 500),    # 
    'PU': range(2200, 4701, 500),
    'TT':range(3000,5501,500) 
    #"TT":range(5000,20001,3000)    # 2200,2700,3200,3700,4200,4700
}
KNN_RANGE = range(10, 61, 10)        # 10,20,30,40,50,60

def run_experiment(dataname, n_sp, knn):
    """执行单个实验并返回结果"""
    command = [
        'python', 'main.py',
        '--dataname', dataname,
        '--n_sp', str(n_sp),
        '--knn', str(knn),
        '--gpu', '0',
        '--seed', '42',
        '--epochs', '40',
        '--patience', '20'
    ]
    
    start_time = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    run_time = time.time() - start_time
    
    # 从输出中解析指标
    output = result.stdout
    metrics = {
        'ACC': extract_metric(output, r'ACC:([\d.]+)'),
        'Kappa': extract_metric(output, r'Kappa:([\d.]+)'),  
        'NMI': extract_metric(output, r'NMI:([\d.]+)'),
        'ARI': extract_metric(output, r'ARI:([\d.]+)')  
    }
    
    return {
        **metrics,
        'runtime': run_time
    }

def extract_metric(output, pattern):
    """使用正则表达式提取指标"""
    import re
    match = re.search(pattern, output)
    return float(match.group(1)) if match else 0.0

def main():
    for dataname in DATASETS:
        csv_file = f'{dataname}_results22.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # writer.writerow(['Timestamp', 'Dataset', 'n_sp', 'knn', 
            #                'ACC', 'NMI', 'F1', 'Runtime(s)'])
            writer.writerow(['Dataset', 'M', 'knn', 
                           'ACC', 'Kappa','NMI', 'ARI', 'Runtime(s)'])
            
            for n_sp in N_SP_RANGES[dataname]:
                for knn in KNN_RANGE:
                    print(f'Running {dataname} - n_sp:{n_sp} knn:{knn}')
                    results = run_experiment(dataname, n_sp, knn)
                    
                    writer.writerow([
                        #datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        dataname,
                        n_sp,
                        knn,
                        results['ACC'],
                        results['Kappa'],
                        results['NMI'],
                        results['ARI'],
                        round(results['runtime'], 2)
                    ])
                    f.flush()  # 实时写入

if __name__ == "__main__":
    main()