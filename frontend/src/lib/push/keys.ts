/**
 * VAPID key conversion.
 *
 * `PushManager.subscribe()` wants the application server key as raw bytes,
 * and every server hands it out as base64url text. This is the standard
 * fifteen lines that bridge the two: base64url's `-`/`_` back to `+`/`/`,
 * the padding the encoding drops back on, then bytes.
 */

/** base64url text → the bytes `applicationServerKey` expects. */
export function urlBase64ToUint8Array(b64url: string): Uint8Array {
	const trimmed = b64url.trim();
	const padding = '='.repeat((4 - (trimmed.length % 4)) % 4);
	const base64 = (trimmed + padding).replace(/-/g, '+').replace(/_/g, '/');
	const raw = atob(base64);
	const bytes = new Uint8Array(raw.length);
	for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
	return bytes;
}

/**
 * A shape check, not a validation: an uncompressed P-256 point is 65 bytes
 * starting with 0x04. Used to fail early with a clear reason rather than
 * inside the browser's `subscribe()`.
 */
export function isPlausibleVapidKey(b64url: string): boolean {
	try {
		const bytes = urlBase64ToUint8Array(b64url);
		return bytes.length === 65 && bytes[0] === 0x04;
	} catch {
		return false;
	}
}
