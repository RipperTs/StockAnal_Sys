import hashlib

def md5_encrypt(text, encoding='utf-8', output_format='hex'):
    """
    对字符串进行MD5加密
    
    参数:
        text: 要加密的字符串或字节
        encoding: 字符串编码方式，默认为utf-8
        output_format: 输出格式，'hex'表示十六进制字符串，'bytes'表示字节
        
    返回:
        根据output_format返回相应格式的MD5哈希值
    """
    if isinstance(text, str):
        text = text.encode(encoding)
    
    md5_hash = hashlib.md5(text)
    
    if output_format == 'hex':
        return md5_hash.hexdigest()
    elif output_format == 'bytes':
        return md5_hash.digest()
    else:
        raise ValueError("不支持的输出格式，请使用'hex'或'bytes'")
