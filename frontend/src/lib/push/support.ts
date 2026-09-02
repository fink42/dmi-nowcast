/**
 * Can this browser, on this device, in this window, receive a push message?
 *
 * Four answers rather than a boolean, because the three failing ones need
 * three different sentences. The one that matters is iOS: Safari has
 * supported Web Push since 16.4, but *only* for a site the user added to the
 * home screen. A plain Safari tab exposes no `PushManager` at all, which is
 * indistinguishable from "this browser cannot do push" unless the UA is
 * inspected — and telling an iPhone owner their browser does not support
 * notifications, when one Share-sheet tap would fix it, is the worst answer
 * of the four.
 *
 * The environment is injected so every branch is testable; `currentEnv()` is
 * the only part that touches globals, and it never throws.
 */

export type PushSupport = 'supported' | 'unsupported' | 'ios-not-installed' | 'insecure-context';

export interface PushEnv {
	hasServiceWorker: boolean;
	hasPushManager: boolean;
	hasNotification: boolean;
	isSecureContext: boolean;
	userAgent: string;
	/** Running as an installed app (iOS `navigator.standalone` or display-mode). */
	standalone: boolean;
	/** iPadOS 13+ reports a desktop Safari UA; touch points give it away. */
	maxTouchPoints: number;
}

const IOS_UA = /iPhone|iPad|iPod/i;
const MAC_UA = /Macintosh|Mac OS X/i;

/**
 * iPhone, iPad or iPod — including an iPad pretending to be a Mac, which it
 * does by default since iPadOS 13. A real Mac reports no touch points; an
 * iPad reports five.
 */
export function isIosLike(userAgent: string, maxTouchPoints: number): boolean {
	if (IOS_UA.test(userAgent)) return true;
	return MAC_UA.test(userAgent) && maxTouchPoints > 1;
}

export function detectPushSupport(env: PushEnv): PushSupport {
	// Nothing works off https (localhost excepted, which the browser already
	// counts as secure), and no other message would be actionable.
	if (!env.isSecureContext) return 'insecure-context';

	if (isIosLike(env.userAgent, env.maxTouchPoints)) {
		// Deliberately ahead of the capability check: in a plain iOS tab the
		// APIs are absent *because* the site is not installed, and saying
		// "unsupported" there would be both true and useless.
		if (!env.standalone) return 'ios-not-installed';
	}

	if (env.hasServiceWorker && env.hasPushManager && env.hasNotification) return 'supported';
	return 'unsupported';
}

/** Whether the page is running as an installed app. Never throws. */
function readStandalone(): boolean {
	try {
		// iOS only, and non-standard — hence the cast.
		const iosStandalone = (navigator as Navigator & { standalone?: boolean }).standalone;
		if (iosStandalone === true) return true;
	} catch {
		/* fall through to the media query */
	}
	try {
		return window.matchMedia('(display-mode: standalone)').matches;
	} catch {
		return false;
	}
}

/** The real environment, read defensively. Safe to call on the server. */
export function currentEnv(): PushEnv {
	const unknown: PushEnv = {
		hasServiceWorker: false,
		hasPushManager: false,
		hasNotification: false,
		isSecureContext: false,
		userAgent: '',
		standalone: false,
		maxTouchPoints: 0
	};
	try {
		if (typeof window === 'undefined' || typeof navigator === 'undefined') return unknown;
		return {
			hasServiceWorker: 'serviceWorker' in navigator,
			hasPushManager: 'PushManager' in window,
			hasNotification: 'Notification' in window,
			isSecureContext: window.isSecureContext === true,
			userAgent: navigator.userAgent ?? '',
			standalone: readStandalone(),
			maxTouchPoints: Number(navigator.maxTouchPoints ?? 0)
		};
	} catch {
		return unknown;
	}
}
