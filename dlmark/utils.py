from subprocess import check_output

def nv_gpu_mem_usage(dev_id=0):
    ret = check_output([
        'nvidia-smi', '-i', str(dev_id), '-q', '-d', 'MEMORY'])
    lines = [l.split() for l in str(ret).split('\\n') if 'Used' in l]
    assert len(lines) > 0
    assert len(lines[0]) > 3
    return int(lines[0][2])

def nv_gpu_name(dev_id=0):
    ret = check_output(['nvidia-smi', '-i', str(dev_id), '-q'])
    lines = [l.split(':') for l in str(ret).split('\\n')
             if 'Product Name' in l]
    assert len(lines) > 0
    return lines[0][1].strip()
