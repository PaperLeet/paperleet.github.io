export const MAX_REVIEW_USER_MESSAGE=500;
export function validReviewUserMessage(value:unknown):value is string{
 return typeof value==='string'&&Array.from(value).length<=MAX_REVIEW_USER_MESSAGE&&!value.includes('\0');
}
