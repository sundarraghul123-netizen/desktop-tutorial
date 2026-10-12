import re

PAT = re.compile(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})")

def video_id(url):
    m = PAT.search(url or "")
    if not m:
        raise ValueError("Invalid YouTube URL. Example: https://www.youtube.com/watch?v=XXXXXXXXXXX")
    return m.group(1)

def fetch(url):
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        IpBlocked,
        NoTranscriptFound,
        RequestBlocked,
        TranscriptsDisabled,
        VideoUnavailable,
        YouTubeRequestFailed,
    )

    vid = video_id(url)
    api = YouTubeTranscriptApi()
    try:
        t = api.fetch(vid, languages=["en"])
    except NoTranscriptFound as original_error:
        try:
            available = list(api.list(vid))
        except TranscriptsDisabled as e:
            raise RuntimeError(
                "Captions are disabled for this video. Enable them on YouTube or upload "
                "a transcript file (.txt, .vtt, or .srt)."
            ) from e
        except (RequestBlocked, IpBlocked) as e:
            raise RuntimeError(
                "YouTube blocked the caption request from this network. Try again later "
                "or upload a transcript file (.txt, .vtt, or .srt)."
            ) from e
        except VideoUnavailable as e:
            raise RuntimeError(
                "This YouTube video is unavailable. Check that the URL is correct and "
                "that the video is public."
            ) from e
        except YouTubeRequestFailed as e:
            raise RuntimeError(
                "YouTube could not be reached to retrieve captions. Check your connection "
                "and retry, or upload a transcript file."
            ) from e
        except Exception as list_error:
            raise RuntimeError(
                "YouTube has no English captions, and the available caption languages "
                "could not be checked. Verify the video is public and try again, or "
                "upload a transcript file."
            ) from list_error

        if not available:
            raise RuntimeError(
                "This video has no captions available. Enable captions on YouTube or "
                "upload a transcript file (.txt, .vtt, or .srt)."
            ) from original_error

        transcript = next((item for item in available if item.is_translatable), None)
        try:
            if transcript is not None:
                try:
                    t = transcript.translate("en").fetch()
                except Exception:
                    t = transcript.fetch()
            else:
                transcript = available[0]
                t = transcript.fetch()
        except Exception as fallback_error:
            raise RuntimeError(
                "English captions are unavailable and YouTube could not fetch or "
                "translate the available captions. Upload a transcript file instead."
            ) from fallback_error
    except TranscriptsDisabled as e:
        raise RuntimeError(
            "Captions are disabled for this video. Enable them on YouTube or upload "
            "a transcript file (.txt, .vtt, or .srt)."
        ) from e
    except (RequestBlocked, IpBlocked) as e:
        raise RuntimeError(
            "YouTube blocked the caption request from this network. Try again later "
            "or upload a transcript file (.txt, .vtt, or .srt)."
        ) from e
    except VideoUnavailable as e:
        raise RuntimeError(
            "This YouTube video is unavailable. Check that the URL is correct and "
            "that the video is public."
        ) from e
    except YouTubeRequestFailed as e:
        raise RuntimeError(
            "YouTube could not be reached to retrieve captions. Check your connection "
            "and retry, or upload a transcript file."
        ) from e
    segs = [{"start": s.start, "end": s.start + s.duration, "text": s.text} for s in t]
    return segs, getattr(t, "language_code", "")
