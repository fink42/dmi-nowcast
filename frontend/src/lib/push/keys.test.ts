/**
 * The VAPID conversion, against fixed vectors. The interesting inputs are the
 * two base64url substitutions and the stripped padding — get either wrong and
 * `subscribe()` fails in the browser with an opaque error.
 */
import { describe, expect, it } from 'vitest';
import { isPlausibleVapidKey, urlBase64ToUint8Array } from './keys';

describe('urlBase64ToUint8Array', () => {
	it('decodes a padded-length string', () => {
		// 'AQID' is exactly four characters: no padding was ever dropped.
		expect([...urlBase64ToUint8Array('AQID')]).toEqual([1, 2, 3]);
	});

	it('restores the padding base64url drops', () => {
		// 0xff 0xfe → '//4=' in base64; base64url writes '__4'.
		expect([...urlBase64ToUint8Array('__4')]).toEqual([255, 254]);
		// One byte: 'AQ' stands for 'AQ=='.
		expect([...urlBase64ToUint8Array('AQ')]).toEqual([1]);
	});

	it('maps - and _ back to + and /', () => {
		// 0xfb 0xef → '++8=' in base64, '--8' in base64url.
		expect([...urlBase64ToUint8Array('--8')]).toEqual([251, 239]);
		expect([...urlBase64ToUint8Array('__8')]).toEqual([255, 255]);
	});

	it('round-trips a full 65-byte P-256 public key', () => {
		const bytes = new Uint8Array(65);
		bytes[0] = 0x04;
		for (let i = 1; i < 65; i++) bytes[i] = (i * 37 + 11) % 256;
		const b64url = btoa(String.fromCharCode(...bytes))
			.replace(/\+/g, '-')
			.replace(/\//g, '_')
			.replace(/=+$/, '');
		const decoded = urlBase64ToUint8Array(b64url);
		expect(decoded.length).toBe(65);
		expect([...decoded]).toEqual([...bytes]);
	});

	it('ignores surrounding whitespace', () => {
		expect([...urlBase64ToUint8Array('  AQID\n')]).toEqual([1, 2, 3]);
	});
});

describe('isPlausibleVapidKey', () => {
	it('accepts a 65-byte uncompressed point', () => {
		const bytes = new Uint8Array(65).fill(7);
		bytes[0] = 0x04;
		const b64url = btoa(String.fromCharCode(...bytes))
			.replace(/\+/g, '-')
			.replace(/\//g, '_')
			.replace(/=+$/, '');
		expect(isPlausibleVapidKey(b64url)).toBe(true);
		expect(b64url.startsWith('BA')).toBe(true);
	});

	it('rejects the wrong length, the wrong prefix and outright garbage', () => {
		expect(isPlausibleVapidKey('AQID')).toBe(false);
		const compressed = new Uint8Array(65).fill(1);
		const b64url = btoa(String.fromCharCode(...compressed)).replace(/=+$/, '');
		expect(isPlausibleVapidKey(b64url)).toBe(false);
		expect(isPlausibleVapidKey('not base64 at all!!')).toBe(false);
	});
});
