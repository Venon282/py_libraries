import time
import json
import logging
import requests
from PIL import Image
from enum import Enum
from urllib.parse import quote_plus

class MediaType(Enum):
    IMAGE       = "IMAGE"
    VIDEO       = "VIDEO"
    REELS       = "REELS"
    STORIES     = "STORIES"
    CAROUSEL    = "CAROUSEL"

class InstagramError(Exception):
    pass

class Instagram:
    """
    Lightweight wrapper for Instagram Graph API content publishing + basic moderation/insights.
    - Requires an Instagram Professional account (Business/Creator) linked to a Facebook Page.
    - For resumable video uploads (reels/large videos) rupload.facebook.com is used.
    Docs: Content publishing, resumable uploads, media_publish, container status, limits.
    (See Meta docs for current details / permissions / rate limits.)
    """

    def __init__(self, access_token: str, ig_account_id: str, api_version: str = "v23.0",
                 graph_host: str = "https://graph.facebook.com", rupload_host: str = "https://rupload.facebook.com"):
        self.session = requests.Session()
        self.access_token = access_token
        self.ig_id = ig_account_id
        self.api_version = api_version.strip("/")
        self.graph_host = graph_host.rstrip("/")
        self.rupload_host = rupload_host.rstrip("/")
        self.base = f"{self.graph_host}/{self.api_version}"
        # default headers
        self.session.headers.update({
            "Accept": "application/json",
        })
        
    def publishImage(self, image_url:str, caption:str, alt_text:str, timeout_seconds:int=60):
        resp = self.createMediaContainer(image_url=image_url, caption=caption, alt_text=alt_text)
        container_id = resp.get("id")
        
        # optional: check readiness (images are usually quick)
        self.pollContainerUntilReady(container_id, timeout_seconds=timeout_seconds)
        publish_resp = self.publishMedia(container_id)
        logging.info(f"Published media id: {publish_resp}")
        return True

    def _req(self, method, path, params=None, json_data=None, headers=None, timeout=60, host=None):
        """Internal helper for GET/POST/DELETE requests."""
        if host is None:
            host = self.base
        url = f"{host.rstrip('/')}/{path.lstrip('/')}"
        if params is None:
            params = {}
        params["access_token"] = self.access_token
        resp = self.session.request(method, url, params=params, json=json_data, headers=headers, timeout=timeout)
        try:
            data = resp.json()
        except ValueError:
            raise InstagramError(f"Invalid JSON response {resp.status_code}: {resp.text}")
        if resp.status_code >= 400:
            raise InstagramError(f"API error {resp.status_code}: {data}")
        return data

    # -----------------------
    # Content publishing flows
    # -----------------------
    def createMediaContainer(self, image_url: str = None, video_url: str = None,
                               caption: str = None, media_type: MediaType = None,
                               is_carousel_item: bool = False, upload_type: str = None,
                               children: list = None, alt_text: str = None, extra_fields: dict = None):
        """
        Create a media container on /{ig_id}/media.
        - image_url or video_url required for direct non-resumable uploads
        - media_type: IMAGE (default if image_url), VIDEO, REELS, STORIES, CAROUSEL
        - is_carousel_item: True for children items used to build a carousel
        - upload_type=resumable to create a resumable upload session (for large video files)
        - children: list of container IDs (for CAROUSEL media_type)
        - alt_text: (new field for image posts; use if supported)
        Returns: dict (usually {'id': '<IG_CONTAINER_ID>'})
        """
        path = f"{self.ig_id}/media"
        payload = {}
        if caption:
            payload["caption"] = caption
        if image_url:
            payload["image_url"] = image_url
        if video_url:
            payload["video_url"] = video_url
        if media_type:
            payload["media_type"] = media_type
        if is_carousel_item:
            payload["is_carousel_item"] = "true"
        if upload_type:
            payload["upload_type"] = upload_type
        if children:
            # API expects comma-separated list
            payload["children"] = ",".join(children)
        if alt_text:
            payload["alt_text"] = alt_text
        if extra_fields:
            payload.update(extra_fields)

        return self._req("POST", f"{self.api_version}/{self.ig_id}/media", json_data=payload, host=self.graph_host)
    
    def create_carousel_container(self, children_ids: list, caption: str = None):
        """
        Create a CAROUSEL container from children container ids (up to 10).
        """
        if not (1 <= len(children_ids) <= 10):
            raise ValueError("Carousel children list must contain 1-10 media container IDs.")
        return self.createMediaContainer(media_type=MediaType.CAROUSEL, children=children_ids, caption=caption)

    def publishMedia(self, creation_id: str):
        """
        Publish media: POST /{ig_user_id}/media_publish with creation_id
        On success returns {'id': '<IG_MEDIA_ID>'}
        """
        path = f"{self.ig_id}/media_publish"
        payload = {"creation_id": creation_id}
        return self._req("POST", f"{self.api_version}/{path}", json_data=payload, host=self.graph_host)

    def getContainerStatus(self, container_id: str):
        """
        GET /{ig_container_id}?fields=status_code (and optionally status)
        Returns container status (EXPIRED, ERROR, FINISHED, IN_PROGRESS, PUBLISHED)
        """
        path = f"{container_id}"
        params = {"fields": "status_code,status"}
        return self._req("GET", f"{self.api_version}/{path}", params=params, host=self.graph_host)

    def pollContainerUntilReady(self, container_id: str, timeout_seconds: int = 300, interval: int = 5):
        """
        Poll container status until FINISHED or timeout.
        Returns final status response.
        """
        start = time.time()
        while True:
            status = self.getContainerStatus(container_id)
            st = status.get("status_code") or (status.get("status", {}).get("status_code") if isinstance(status.get("status"), dict) else None)
            if st:
                st = st.upper()
            if st in ("FINISHED", "PUBLISHED"):
                return status
            if st in ("ERROR", "EXPIRED"):
                raise InstagramError(f"Container {container_id} returned status {st}: {status}")
            if (time.time() - start) > timeout_seconds:
                raise InstagramError(f"Timeout waiting for container {container_id} to be ready; last status: {status}")
            time.sleep(interval)

    # -----------------------
    # Resumable upload helpers (videos / reels)
    # -----------------------
    def start_resumable_container(self, media_type: MediaType = MediaType.VIDEO, caption: str = None, upload_type: str = "resumable", extra: dict = None):
        """
        Create a container with upload_type=resumable to start a resumable upload session.
        Returns the created container id.
        """
        payload = {"media_type": media_type, "upload_type": upload_type}
        if caption:
            payload["caption"] = caption
        if extra:
            payload.update(extra)
        return self._req("POST", f"{self.api_version}/{self.ig_id}/media", json_data=payload, host=self.graph_host)

    def upload_resumable_local(self, ig_container_id: str, file_path: str):
        """
        Upload a local file (single-chunk) to rupload.facebook.com.
        For full resumable/chunked upload you should implement chunking with offset headers.
        Example POST:
        POST https://rupload.facebook.com/ig-api-upload/{API_VERSION}/{IG_MEDIA_CONTAINER_ID}
        Headers:
          Authorization: OAuth <ACCESS_TOKEN>
          offset: 0
          file_size: <bytes>
        Body: binary data
        """
        # read file and upload in one request (works if file size is tolerable)
        with open(file_path, "rb") as f:
            data = f.read()
        file_size = len(data)
        upload_path = f"ig-api-upload/{self.api_version}/{ig_container_id}"
        url = f"{self.rupload_host}/{upload_path}"
        headers = {
            "Authorization": f"OAuth {self.access_token}",
            "offset": "0",
            "file_size": str(file_size),
            # rupload sometimes expects Content-Type: application/octet-stream for binary uploads
            "Content-Type": "application/octet-stream"
        }
        resp = self.session.post(url, headers=headers, data=data, timeout=120)
        try:
            return resp.json()
        except ValueError:
            # rupload sometimes returns plain text success strings; handle generically
            if resp.status_code == 200:
                return {"success": True, "http_text": resp.text}
            raise InstagramError(f"Upload error: {resp.status_code} -> {resp.text}")

    def upload_resumable_remote(self, ig_container_id: str, public_file_url: str):
        """
        Instruct rupload to fetch a hosted file:
        POST https://rupload.facebook.com/ig-api-upload/{API_VERSION}/{IG_MEDIA_CONTAINER_ID}
        with header: file_url: https://...
        """
        upload_path = f"ig-api-upload/{self.api_version}/{ig_container_id}"
        url = f"{self.rupload_host}/{upload_path}"
        headers = {
            "Authorization": f"OAuth {self.access_token}",
            "file_url": public_file_url
        }
        resp = self.session.post(url, headers=headers, timeout=120)
        try:
            return resp.json()
        except ValueError:
            if resp.status_code == 200:
                return {"success": True, "http_text": resp.text}
            raise InstagramError(f"Upload (remote) error: {resp.status_code} -> {resp.text}")

    # -----------------------
    # Limits, insights, basic media fetch
    # -----------------------
    def getContentPublishingLimit(self):
        """GET /{ig_id}/content_publishing_limit"""
        return self._req("GET", f"{self.api_version}/{self.ig_id}/content_publishing_limit", host=self.graph_host)

    def get_media(self, media_id: str, fields: str = "id,media_type,media_url,caption,permalink"):
        return self._req("GET", f"{self.api_version}/{media_id}", params={"fields": fields}, host=self.graph_host)

    def get_media_insights(self, media_id: str, metrics: list):
        """GET /{ig_media_id}/insights?metric=impressions,reach,..."""
        metrics_str = ",".join(metrics)
        return self._req("GET", f"{self.api_version}/{media_id}/insights", params={"metric": metrics_str}, host=self.graph_host)

    def get_user_insights(self, metrics: list, period: str = "lifetime"):
        """GET /{ig_id}/insights?metric=...&period=..."""
        metrics_str = ",".join(metrics)
        return self._req("GET", f"{self.api_version}/{self.ig_id}/insights", params={"metric": metrics_str, "period": period}, host=self.graph_host)

    # -----------------------
    # Comment moderation
    # -----------------------
    def get_comments(self, media_id: str, params: dict = None):
        """GET /{ig_media_id}/comments"""
        if params is None:
            params = {}
        return self._req("GET", f"{self.api_version}/{media_id}/comments", params=params, host=self.graph_host)

    def create_comment(self, media_id: str, message: str):
        """POST /{ig_media_id}/comments?message=..."""
        return self._req("POST", f"{self.api_version}/{media_id}/comments", json_data={"message": message}, host=self.graph_host)

    def reply_to_comment(self, comment_id: str, message: str):
        """POST /{ig_comment_id}/replies?message=..."""
        return self._req("POST", f"{self.api_version}/{comment_id}/replies", json_data={"message": message}, host=self.graph_host)

    def hide_comment(self, comment_id: str, hide: bool = True):
        """POST /{ig_comment_id}?is_hidden=true|false"""
        return self._req("POST", f"{self.api_version}/{comment_id}", json_data={"is_hidden": hide}, host=self.graph_host)

    def delete_comment(self, comment_id: str):
        """DELETE /{ig_comment_id} (delete a comment)"""
        return self._req("DELETE", f"{self.api_version}/{comment_id}", host=self.graph_host)

    # -----------------------
    # Optional helper: upload local image to S3 (requires boto3)
    # -----------------------
    def upload_file_to_s3(self, file_path: str, bucket: str, key: str, acl: str = "public-read", boto3_session=None):
        """
        Helper: upload file to S3 so it becomes publicly accessible (useful because image_url must be public).
        Requires boto3. Returns the public URL (assuming bucket is public).
        """
        try:
            import boto3
        except ImportError:
            raise RuntimeError("boto3 is required for upload_file_to_s3; install boto3 or host your image elsewhere.")

        s3 = boto3_session or boto3.client("s3")
        with open(file_path, "rb") as f:
            s3.put_object(Bucket=bucket, Key=key, Body=f, ACL=acl, ContentType="image/jpeg")
        # example public URL pattern - adjust if you use different bucket hosting / cloudfront
        return f"https://{bucket}.s3.amazonaws.com/{quote_plus(key)}"

     
    def getLongLifeUserAccessToken(self, app_id, app_secret, short_lived_token):
        """
        Interrogez le point de terminaison GET oauth/access_token.

        curl -i -X GET "https://graph.facebook.com/{graph-api-version}/oauth/access_token?  
            grant_type=fb_exchange_token&          
            client_id={app-id}&
            client_secret={app-secret}&
            fb_exchange_token={your-access-token}" 
        Exemple de réponse
        {
        "access_token":"{long-lived-user-access-token}",
        "token_type": "bearer",
        "expires_in": 5183944            //The number of seconds until the token expires
        }
        """
        params = {
            "grant_type": "fb_exchange_token",
            "client_id": app_id,
            "client_secret": app_secret,
            "fb_exchange_token": short_lived_token
        }
        path = f"{self.api_version}/oauth/access_token"
        
        return self._req("GET", path,  params=params, host=self.graph_host)

    def refreshLongLivedUserAccessToken(self, app_id: str, app_secret: str):
        """
        Refresh the current long-lived user access token.

        Facebook allows re-exchanging an unexpired long-lived token to get a new 60-day token.
        Returns:
        {
            "access_token": "<new-long-lived-token>",
            "token_type": "bearer",
            "expires_in": 5184000
        }
        """
        if not self.access_token:
            raise InstagramError("No access_token available to refresh.")

        params = {
            "grant_type": "fb_exchange_token",
            "client_id": app_id,
            "client_secret": app_secret,
            "fb_exchange_token": self.access_token  # use current long-lived token
        }
        path = f"{self.api_version}/oauth/access_token"
        return self._req("GET", path, params=params, host=self.graph_host)
    
    def fixAspectRatio(self, img_path, min_ratio=0.8, max_ratio=1.91):
        img = Image.open(img_path)
        w, h = img.size
        ratio = w / h
        if ratio < min_ratio:
            # too tall → crop height
            new_h = int(w / min_ratio)
            top = (h - new_h) // 2
            img = img.crop((0, top, w, top + new_h))
        elif ratio > max_ratio:
            # too wide → crop width
            new_w = int(h * max_ratio)
            left = (w - new_w) // 2
            img = img.crop((left, 0, left + new_w, h))
        return img


if __name__ == '__main__':
    ACCESS_TOKEN = "EAAC...LONGTOKEN"
    IG_ACCOUNT_ID = "1784...your_ig_business_id"
    api = Instagram(ACCESS_TOKEN, IG_ACCOUNT_ID, api_version="v17.0")

    # 1) publish a single image (must be on a public URL)
    resp = api.createMediaContainer(image_url="https://cdn.example.com/photo.jpg", caption="Hello from API!", alt_text="A scenic view")
    container_id = resp.get("id")
    # optional: check readiness (images are usually quick)
    api.pollContainerUntilReady(container_id, timeout_seconds=60)
    publish_resp = api.publishMedia(container_id)
    print("Published media id:", publish_resp)

    # 2) publish a carousel:
    # - create containers for each child (is_carousel_item=true),
    c1 = api.createMediaContainer(image_url="https://cdn.example.com/i1.jpg", is_carousel_item=True).get("id")
    c2 = api.createMediaContainer(image_url="https://cdn.example.com/i2.jpg", is_carousel_item=True).get("id")
    carousel = api.create_carousel_container([c1, c2], caption="Carousel post!")
    api.pollContainerUntilReady(carousel.get("id"))
    api.publishMedia(carousel.get("id"))

    # 3) resumable video (reels):
    res = api.start_resumable_container(media_type=MediaType.REELS, caption="A reel")
    container_id = res.get("id")
    # upload local file (or upload_resumable_remote if hosted)
    upload_resp = api.upload_resumable_local(container_id, "/path/to/video.mp4")
    # poll
    api.pollContainerUntilReady(container_id, timeout_seconds=600, interval=10)
    api.publishMedia(container_id)
