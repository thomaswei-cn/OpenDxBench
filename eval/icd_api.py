import time, requests, urllib3
from config import *

from urllib3.exceptions import InsecureRequestWarning

urllib3.disable_warnings(InsecureRequestWarning)

def try_icd_encoding(term):
    """
    尝试使用自动编码和灵活搜索获取ICD编码
    """
    try:
        # 尝试 autocode
        results = search_icd_11(term)
        if results and 'theCode' in results:
            return {
                "code": results['theCode'],
                "title": results['matchingText'],
                "chapter": search_chapter_by_code(results['theCode'])
            }
        else:
            print("icd autocode failed, trying flex search...")
            # 如果失败，尝试 flex search
            flex_name = flex_search_icd_11(term)['label']
            flex_result = search_icd_11(flex_name)
            # print(f"ICD flex lookup for '{flex_name}': {flex_result}")
            if flex_result:
                return {
                    "code": flex_result['theCode'],
                    "title": flex_result['matchingText'],
                    "chapter": search_chapter_by_code(flex_result['theCode'])
                }
    except Exception as e:
        print(f"ICD lookup failed for '{term}': {e}")
    return None

_token_cache = {
    "access_token": None,
    "expiry": 0,
}


SCOPE = 'icdapi_access'
GRANT_TYPE = 'client_credentials'
TOKEN_ENDPOINT = 'https://icdaccessmanagement.who.int/connect/token'


def _get_token():
    """
    获取并缓存 OAuth2 token，自动处理过期。
    """
    now = time.time()
    if not _token_cache["access_token"] or now >= _token_cache["expiry"]:
        payload = {
            "client_id": ICD_CLIENT_ID,
            "client_secret": ICD_CLIENT_SECRET,
            "scope": SCOPE,
            "grant_type": GRANT_TYPE,
        }
        resp = requests.post(TOKEN_ENDPOINT, data=payload, verify=False)
        resp.raise_for_status()
        data = resp.json()
        _token_cache["access_token"] = data["access_token"]
        # expires_in 单位为秒，提前 10s 刷新
        _token_cache["expiry"] = now + data.get("expires_in", 0) - 10
    return _token_cache["access_token"]

def search_icd_11(name):

    token = _get_token()

    # access ICD API
    name = name.replace(' ', '%20')  # URL encode spaces
    uri = f'https://id.who.int/icd/release/11/2025-01/mms/autocode?searchText={name}'

    # HTTP header fields to set
    headers = {'Authorization': 'Bearer ' + token,
               'Accept': 'application/json',
               'Accept-Language': 'en',
               'API-Version': 'v2'}

    # make request
    r = requests.get(uri, headers=headers, verify=False)

    # print the result
    return r.json()


def flex_search_icd_11(name):
    """
    Flexible search for ICD-11 codes, allowing for partial matches.
    """

    token = _get_token()

    # access ICD API
    name = name.replace(' ', '%20')  # URL encode spaces
    uri = f'https://id.who.int/icd/release/11/2025-01/mms/search?q={name}&subtreeFilterUsesFoundationDescendants=false&includeKeywordResult=false&useFlexisearch=true&flatResults=false&highlightingEnabled=false&medicalCodingMode=true'

    # HTTP header fields to set
    headers = {'Authorization': 'Bearer ' + token,
               'Accept': 'application/json',
               'Accept-Language': 'en',
               'API-Version': 'v2'}

    # make request
    r = requests.get(uri, headers=headers, verify=False)
    result = r.json()
    max_pos = 0
    max_score = 0
    if len(result['destinationEntities']) == 0:
        raise ValueError("flex search failed")
    for idx, entity in enumerate(result['destinationEntities']):
        if entity['score'] > max_score:
            max_score = entity['score']
            max_pos = idx
    res = result['destinationEntities'][max_pos]
    # print the result
    return res

def get_url_code(icd_code, headers):
    if "&" in icd_code:
        icd_code = icd_code.split('&')[0]
    uri = f'https://id.who.int/icd/release/11/2025-01/mms/codeinfo/{icd_code}?flexiblemode=false&convertToTerminalCodes=false'
    r = requests.get(uri, headers=headers, verify=False).json()
    return r['stemId'].split('mms/')[-1].split('/')[0]


def get_info_by_url_code(icd_url_code, headers):
    uri = f'https://id.who.int/icd/entity/{icd_url_code}?releaseId=2025-01'
    r = requests.get(uri, headers=headers, verify=False)
    r.raise_for_status()
    r = r.json()
    return r



def search_chapter_by_code(icd_code):
    """
    Search for a chapter by its ICD-11 code.

    :param icd_code: The ICD-11 code to search for.
    :return: The chapter information if found, otherwise None.
    """
    token = _get_token()

    headers = {'Authorization': 'Bearer ' + token,
               'Accept': 'application/json',
               'Accept-Language': 'en',
               'API-Version': 'v2'}

    url_code = get_url_code(icd_code, headers)
    cur_url_info = get_info_by_url_code(url_code, headers)
    while True:
        parent_url_code = cur_url_info['parent'][0].split('/')[-1]
        if parent_url_code == '455013390':
            break
        else:
            cur_url_info = get_info_by_url_code(parent_url_code, headers)
    title = cur_url_info['title']['@value']
    return title

if __name__ == "__main__":
    diagnosis = "Osteochemonecrosis of fibular bone graft"
    res = search_icd_11(diagnosis)
    print(res)
